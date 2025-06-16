# Imports

# > Standard library
import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# > Third-party dependencies
import tensorflow as tf
from bidi.algorithm import get_display

# Adjust path for worker execution
current_worker_file_dir = os.path.dirname(os.path.realpath(__file__))
api_dir = os.path.dirname(current_worker_file_dir)
src_dir = os.path.dirname(api_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# > Local imports
from utils.decoding import decode_batch_predictions  # noqa: E402
from utils.text import Tokenizer  # noqa: E402

logger = logging.getLogger(__name__)


def _create_tokenizer(model_dir: str | os.PathLike[str]) -> Tokenizer:
    """
    Instantiate a :class:`~utils.text.Tokenizer` from *model_dir*.

    The function looks for either a ``tokenizer.json`` (preferred) or a
    ``charlist.txt`` file inside *model_dir* and delegates the actual loading
    to :pymeth:`utils.text.Tokenizer.load_from_file`.

    Parameters
    ----------
    model_dir : str or pathlib.Path
        Directory containing ``tokenizer.json`` or ``charlist.txt``.

    Returns
    -------
    utils.text.Tokenizer
        The initialised tokenizer ready for inference‑time decoding.

    Raises
    ------
    FileNotFoundError
        If neither ``tokenizer.json`` nor ``charlist.txt`` can be found inside
        *model_dir*.
    """
    model_dir = Path(model_dir)
    tok_json = model_dir / "tokenizer.json"
    char_txt = model_dir / "charlist.txt"
    file = tok_json if tok_json.exists() else char_txt
    if not file.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found in {model_dir!s} (expected 'tokenizer.json' "
            "or 'charlist.txt')."
        )

    logger.debug("Tokenizer initialised from %s", file)
    return Tokenizer.load_from_file(file.absolute().as_posix())


def _fetch_metadata_for_batch(
    whitelists_tensor: tf.Tensor | tf.RaggedTensor,
    base_model_dir: str | os.PathLike[str],
    model_name: str,
) -> List[str]:
    """Retrieve metadata for each sample in a batch.

    Metadata are extracted from a ``config.json`` file located at
    ``base_model_dir / model_name / 'config.json'``. The *whitelists_tensor*
    contains—per sample—a list of JSON keys that should be looked up inside the
    configuration. For efficiency, results are cached within a single call.

    Parameters
    ----------
    whitelists_tensor : tf.Tensor or tf.RaggedTensor
        Rank‑2 ragged/regular tensor containing the keys for which metadata is
        required. Each sample corresponds to one *row*.
    base_model_dir : str or pathlib.Path
        Root directory that contains the individual model checkpoints.
    model_name : str
        Name of the model whose configuration should be queried.

    Returns
    -------
    list[str]
        A JSON‑serialised dictionary per batch sample (same outer length as
        *whitelists_tensor*).
    """
    cfg_path = Path(base_model_dir) / model_name / "config.json"
    try:
        cfg_data: Dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf‑8"))
    except FileNotFoundError:
        cfg_data = {}
    except json.JSONDecodeError as exc:  # pragma: no cover – robustness
        logger.error("Invalid JSON in %s: %s", cfg_path, exc)
        cfg_data = {}

    def _recursive_find(d: Dict[str, Any], key: str) -> Any | None:
        """Depth‑first search for *key* inside nested dictionaries."""
        if key in d:
            return d[key]
        for value in d.values():
            if isinstance(value, dict):
                res = _recursive_find(value, key)
                if res is not None:
                    return res
        return None

    # Convert Tensor/RaggedTensor to a list‑of‑lists[str]
    if isinstance(whitelists_tensor, tf.RaggedTensor):
        processed = [
            [k.numpy().decode() for k in whitelists_tensor.row(i) if k.numpy()]
            for i in range(whitelists_tensor.nrows())
        ]
    else:
        processed = [
            [k.numpy().decode() for k in whitelists_tensor[i] if k.numpy()]
            for i in range(tf.shape(whitelists_tensor)[0])
        ]

    out: List[str] = []
    cache: Dict[str, Any] = {}
    for keys in processed:
        meta: Dict[str, Any] = {}
        for key in keys:
            if not key:
                continue
            if key in cache:
                meta[key] = cache[key]
                continue
            value = _recursive_find(cfg_data, key)
            cache[key] = value if value is not None else "NOT_FOUND"
            meta[key] = cache[key]
        out.append(json.dumps(meta))
    return out


def _format_and_enqueue(
    decoded: List[Tuple[float, str]],
    groups: tf.Tensor,
    image_ids: tf.Tensor,
    meta_json: List[str],
    uniq_keys: tf.Tensor,
    out_q: mp.Queue,
    *,
    bidi: bool = False,
) -> int:
    """Format the decoded predictions and push them onto *out_q*.

    Parameters
    ----------
    decoded : list[tuple[float, str]]
        A list where each item is ``(confidence, text)``.
    groups : tf.Tensor
        Tensor containing *group IDs* for each sample.
    image_ids : tf.Tensor
        Tensor containing unique *image IDs* for each sample.
    meta_json : list[str]
        Pre‑fetched metadata JSON strings corresponding to each sample.
    uniq_keys : tf.Tensor
        Tensor with unique keys that allow downstream consumers to correctly
        merge results.
    out_q : multiprocessing.Queue
        The queue to which formatted results are written.
    bidi : bool, optional
        If *True*, the text is re‑shaped for correct bidirectional display
        using :pymeth:`bidi.algorithm.get_display`.

    Returns
    -------
    int
        Number of successfully enqueued items.
    """
    sent = 0
    for i, (conf, txt) in enumerate(decoded):
        try:
            gid = groups[i].numpy().decode("utf‑8", "ignore")
            iid = image_ids[i].numpy().decode("utf‑8", "ignore")
            meta = json.loads(meta_json[i])
            ukey = uniq_keys[i].numpy().decode("utf‑8", "ignore")
        except Exception as exc:
            logger.error("Error decoding batch item metadata: %s", exc)
            try:
                # Attempt to send an error back to the client
                ukey = uniq_keys[i].numpy().decode("utf-8", "ignore")
                gid = groups[i].numpy().decode("utf-8", "ignore")
                iid = image_ids[i].numpy().decode("utf-8", "ignore")
                error_payload = {
                    "group_id": gid,
                    "identifier": iid,
                    "error": "ResultDecodingFailed",
                    "detail": f"An unexpected error occurred while processing result: {exc}",
                }
                out_q.put((error_payload, ukey), timeout=5.0)
            except Exception as send_exc:
                logger.critical(
                    "Failed to send decoding error back to client: %s", send_exc
                )
            continue

        if not iid and not gid:
            continue  # padding

        if bidi:
            txt = get_display(txt)

        result_str = "\t".join([iid, json.dumps(meta), f"{conf}", txt])
        try:
            out_q.put(
                ({"group_id": gid, "identifier": iid, "result": result_str}, ukey),
                timeout=5.0,
            )
            sent += 1
        except mp.queues.Full:  # type: ignore[attr-defined]
            logger.error("Final queue full. Dropping %s", iid)
        except Exception as exc:
            logger.error("Error enqueuing result for %s: %s", iid, exc)
    return sent


def _process_batch(
    batch: Tuple[Any, ...],
    tokenizer: Tokenizer,
    base_model_dir: str | os.PathLike[str],
    current_model: str,
    out_q: mp.Queue,
) -> int:
    """Decode a single batch and enqueue the formatted results.

    This helper merely unpacks *batch*, runs :func:`decode_batch_predictions`,
    fetches the corresponding metadata, and finally delegates to
    :func:`_format_and_enqueue`.

    Parameters
    ----------
    batch : tuple
        Tuple containing ``(enc_preds, groups_t, ids_t, model_name, batch_uuid,
        whitelists_t, uniq_keys_t)`` – see the calling context.
    tokenizer : utils.text.Tokenizer
        Tokenizer instance used to convert model outputs into strings.
    base_model_dir : str or pathlib.Path
        Root directory that contains all model checkpoints.
    current_model : str
        Name of the model currently in use; affects tokenizer reload logic.
    out_q : multiprocessing.Queue
        Queue used to transfer results back to the main process.

    Returns
    -------
    int
        Number of items that were successfully enqueued.
    """
    (
        enc_preds,
        groups_t,
        ids_t,
        model_name,
        batch_uuid,
        whitelists_t,
        uniq_keys_t,
    ) = batch

    decoded = decode_batch_predictions(enc_preds, tokenizer)
    meta = _fetch_metadata_for_batch(whitelists_t, base_model_dir, model_name)
    sent = _format_and_enqueue(decoded, groups_t, ids_t, meta, uniq_keys_t, out_q)
    return sent


def _run_decoder_loop(
    in_q: mp.Queue,
    out_q: mp.Queue,
    cfg: Dict[str, str],
    stop_event: mp.Event,
) -> None:
    """Main processing loop for the decoder worker.

    The loop reads batches from *in_q*, decodes them, and writes the formatted
    results to *out_q*. It exits gracefully when *stop_event* is set or when a
    sentinel ``None`` is received.

    Parameters
    ----------
    in_q : multiprocessing.Queue
        Queue from which encoded prediction batches are read.
    out_q : multiprocessing.Queue
        Queue to which final (decoded) results are written.
    cfg : dict[str, str]
        Mapping with at least ``{"base_model_dir": str, "model_name": str}``.
    stop_event : multiprocessing.Event
        Event used to coordinate a graceful shutdown.
    """

    tf.config.set_visible_devices([], "GPU")  # CPU‑only for decoder

    current_model = cfg["model_name"]
    tokenizer = _create_tokenizer(os.path.join(cfg["base_model_dir"], current_model))

    total = 0
    while not stop_event.is_set():
        try:
            batch = in_q.get(timeout=0.1)
        except mp.queues.Empty:  # type: ignore[attr-defined]
            continue
        if batch is None:
            logger.info("Decoder received sentinel; exiting loop.")
            break

        batch_model = batch[3]  # model_name field
        if batch_model != current_model:
            try:
                tokenizer = _create_tokenizer(
                    os.path.join(cfg["base_model_dir"], batch_model)
                )
                logger.info("Tokenizer switched: %s → %s", current_model, batch_model)
                current_model = batch_model
            except Exception as exc:
                logger.error("Tokenizer reload failed (%s); skipping batch", exc)
                continue

        try:
            sent = _process_batch(
                batch, tokenizer, cfg["base_model_dir"], current_model, out_q
            )
            if sent:
                total += sent
                logger.debug("Batch %s: %d items sent; total=%d", batch[4], sent, total)
        except Exception as exc:
            logger.critical(
                "Decoder failed to process batch %s: %s", batch[4], exc, exc_info=True
            )
            # Unpack the batch to send error messages to all affected clients
            (
                _,
                groups_t,
                ids_t,
                _,
                _,
                _,
                uniq_keys_t,
            ) = batch
            for i in range(tf.shape(ids_t)[0]):
                try:
                    gid = groups_t[i].numpy().decode("utf-8", "ignore")
                    iid = ids_t[i].numpy().decode("utf-8", "ignore")
                    ukey = uniq_keys_t[i].numpy().decode("utf-8", "ignore")
                    if not iid and not gid:
                        continue  # Skip padding
                    error_payload = {
                        "group_id": gid,
                        "identifier": iid,
                        "error": "BatchDecodingFailed",
                        "detail": f"The processing batch failed with an unexpected error: {exc}",
                    }
                    out_q.put((error_payload, ukey), timeout=5.0)
                except Exception as send_exc:
                    logger.critical(
                        "Failed to send batch-level decoding error to client: %s",
                        send_exc,
                    )

    logger.info("Decoder worker exiting. Total processed=%d", total)


def decoder_process_entrypoint(
    mp_predicted_batches_queue: mp.Queue,
    mp_final_results_queue: mp.Queue,
    config: Dict[str, Any],
    stop_event: mp.Event,
) -> None:
    """Entry‑point for the decoder worker process.

    This thin wrapper converts *config* into the subset required by the
    internal helpers, logs start‑up information, and delegates control to
    :func:`_run_decoder_loop`.

    Parameters
    ----------
    mp_predicted_batches_queue : multiprocessing.Queue
        Queue originating from the predictor process. Contains encoded batches.
    mp_final_results_queue : multiprocessing.Queue
        Queue to which fully decoded and formatted results are pushed.
    config : dict[str, Any]
        Application configuration containing at least ``base_model_dir`` and
        ``model_name`` keys.
    stop_event : multiprocessing.Event
        Event that allows the main process to signal termination.
    """
    worker_config = {
        "base_model_dir": config["base_model_dir"],
        "model_name": config["model_name"],
    }
    logger.info(f"Decoder worker starting with config: {worker_config}")

    try:
        _run_decoder_loop(
            mp_predicted_batches_queue,
            mp_final_results_queue,
            worker_config,
            stop_event,
        )
    except Exception as exc:
        logger.critical("Decoder loop crashed: %s", exc, exc_info=True)
    finally:
        logger.info("Decoder worker shut down.")

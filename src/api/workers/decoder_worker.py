# Imports

# > Standard library
import json
import logging
import multiprocessing as mp
import os
import sys
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


def _create_tokenizer(model_dir: str) -> Tokenizer:
    tok_json = os.path.join(model_dir, "tokenizer.json")
    char_txt = os.path.join(model_dir, "charlist.txt")
    file = tok_json if os.path.exists(tok_json) else char_txt
    if not os.path.exists(file):
        raise FileNotFoundError(f"Tokenizer file not found in {model_dir}")
    logger.debug("Tokenizer initialised from %s", file)
    return Tokenizer.load_from_file(file)


def _fetch_metadata_for_batch(
    whitelists_tensor: tf.Tensor,
    base_model_dir: str,
    model_name: str,
) -> List[str]:
    config_path = os.path.join(base_model_dir, model_name, "config.json")
    try:
        with open(config_path, "r", encoding="utf‑8") as f:
            cfg_data = json.load(f)
    except FileNotFoundError:
        cfg_data = {}
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", config_path, exc)
        cfg_data = {}

    def _find_key(d: Dict[str, Any], key: str):
        if key in d:
            return d[key]
        for v in d.values():
            if isinstance(v, dict):
                res = _find_key(v, key)
                if res is not None:
                    return res
        return None

    processed: List[List[str]]
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
        for k in keys:
            if not k:
                continue
            if k in cache:
                meta[k] = cache[k]
                continue
            v = _find_key(cfg_data, k)
            cache[k] = v if v is not None else "NOT_FOUND"
            meta[k] = cache[k]
        out.append(json.dumps(meta))
    return out


def _format_and_enqueue(
    decoded: List[Tuple[float, str]],
    groups: tf.Tensor,
    image_ids: tf.Tensor,
    meta_json: List[str],
    uniq_keys: tf.Tensor,
    out_q: mp.Queue,
    bidi: bool = False,
) -> int:
    sent = 0
    for i, (conf, txt) in enumerate(decoded):
        try:
            gid = groups[i].numpy().decode("utf‑8", "ignore")
            iid = image_ids[i].numpy().decode("utf‑8", "ignore")
            meta = json.loads(meta_json[i])
            ukey = uniq_keys[i].numpy().decode("utf‑8", "ignore")
        except Exception as exc:
            logger.error("Identifier decode error: %s", exc)
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
    base_model_dir: str,
    current_model: str,
    out_q: mp.Queue,
) -> int:
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
):
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

        sent = _process_batch(
            batch, tokenizer, cfg["base_model_dir"], current_model, out_q
        )
        if sent:
            total += sent
            logger.debug("Batch %s: %d items sent; total=%d", batch[4], sent, total)

    logger.info("Decoder worker exiting. Total processed=%d", total)


def decoder_process_entrypoint(
    mp_predicted_batches_queue: mp.Queue,
    mp_final_results_queue: mp.Queue,
    config: Dict[str, Any],
    stop_event: mp.Event,
):
    """
    Entry point for the decoder worker process.

    Parameters
    ----------
    mp_predicted_batches_queue : mp.Queue
        Queue from predictor process containing encoded batches.
    mp_final_results_queue : mp.Queue
        Queue to which final decoded results should be pushed.
    config : dict
        App configuration with base_model_dir and model_name.
    stop_event : mp.Event
        Event to signal termination.
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

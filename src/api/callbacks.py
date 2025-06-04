import logging
import time

import httpx


def attempt_callback(callback_url, payload):
    MAX_RETRIES = 3  # total attempts (initial call + 2 more)
    BACKOFF_BASE = 2  # seconds; grows as 2, 4, 8…
    try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                httpx.post(callback_url, json=payload, timeout=10)
                logging.debug(
                    "Callback succeeded on attempt %d to %s", attempt, callback_url
                )
                break  # success → exit loop

            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                # Only retry on network-level / 5xx issues.
                logging.warning("Callback attempt %d failed: %s", attempt, e)

                if attempt == MAX_RETRIES:
                    logging.error("Exhausted retries. Raising last exception.")
                    raise  # bubble up after final failed attempt

                # back-off before the next try
                sleep_for = BACKOFF_BASE ** (attempt - 1)
                logging.debug("Sleeping %.1fs before retry", sleep_for)
                time.sleep(sleep_for)

    except Exception as e:
        logging.error("Failed to make callback after retries. Error: %s", e)
        raise

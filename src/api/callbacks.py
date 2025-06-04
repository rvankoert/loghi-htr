import logging
import time
import httpx  # Make sure this is in requirements.txt

# Max retries for the callback attempt
MAX_CALLBACK_RETRIES = 3
# Base for exponential backoff, e.g., 2 means 1s, 2s, 4s for attempts
CALLBACK_BACKOFF_BASE = 2  # seconds
# Timeout for each individual HTTP POST request for callback
CALLBACK_TIMEOUT = 10  # seconds


def attempt_callback(callback_url: str, payload: dict):
    """
    Attempt to send a POST request to the callback_url with JSON payload.
    Includes retries with exponential backoff for transient network errors or 5xx server errors.

    Parameters
    ----------
    callback_url : str
        The URL to send the POST request to.
    payload : dict
        The JSON payload for the request.

    Raises
    ------
    Exception
        If all retry attempts fail, the last exception is re-raised.
    """
    logger = logging.getLogger(__name__)

    if not callback_url or callback_url.lower() == "none":
        logger.debug(
            "Callback URL is not provided or set to 'none'. Skipping callback."
        )
        return

    for attempt in range(1, MAX_CALLBACK_RETRIES + 1):
        try:
            response = httpx.post(callback_url, json=payload, timeout=CALLBACK_TIMEOUT)
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses
            logger.debug(
                "Callback succeeded on attempt %d to %s. Status: %d",
                attempt,
                callback_url,
                response.status_code,
            )
            return  # Success, exit function

        except httpx.TimeoutException as e:
            logger.warning(
                "Callback attempt %d to %s timed out: %s", attempt, callback_url, e
            )
            # Treat timeout as a potentially transient issue, so retry
        except httpx.HTTPStatusError as e:
            # For 5xx errors, retry. For 4xx, it's a client error, likely won't be fixed by retry.
            if 500 <= e.response.status_code < 600:
                logger.warning(
                    "Callback attempt %d to %s failed with server error %d: %s",
                    attempt,
                    callback_url,
                    e.response.status_code,
                    e,
                )
            else:  # 4xx error
                logger.error(
                    "Callback attempt %d to %s failed with client error %d: %s. No further retries for this error.",
                    attempt,
                    callback_url,
                    e.response.status_code,
                    e,
                )
                raise  # Do not retry 4xx errors, re-raise immediately
        except (
            httpx.RequestError
        ) as e:  # Covers other network issues like DNS resolution, connection refused
            logger.warning(
                "Callback attempt %d to %s failed with network error: %s",
                attempt,
                callback_url,
                e,
            )
        # Any other unexpected exception during callback
        except Exception as e:
            logger.error(
                "Unexpected error on callback attempt %d to %s: %s",
                attempt,
                callback_url,
                e,
            )
            # Depending on policy, might not retry for unexpected errors
            if attempt == MAX_CALLBACK_RETRIES:
                raise
            # Fall through to retry logic for unexpected errors too, for now

        # If this is the last attempt and it failed, log error and then loop will exit, re-raising
        if attempt == MAX_CALLBACK_RETRIES:
            logger.error(
                "Exhausted all %d retries for callback to %s. Last error will be raised.",
                MAX_CALLBACK_RETRIES,
                callback_url,
            )
            raise Exception(
                f"Callback to {callback_url} failed after {MAX_CALLBACK_RETRIES} attempts with an unspecified error."
            )

        # Calculate sleep time for exponential backoff
        sleep_for = CALLBACK_BACKOFF_BASE ** (attempt - 1)  # 2^0, 2^1, 2^2...
        # Add some jitter to avoid thundering herd, max 1 second
        # sleep_for += random.uniform(0, min(sleep_for * 0.1, 1.0))
        logger.debug(
            "Sleeping %.2fs before next callback retry to %s", sleep_for, callback_url
        )
        time.sleep(sleep_for)

    # If loop finishes without returning (i.e. all retries failed and no exception re-raised inside loop)
    # This part should ideally not be reached if exceptions are re-raised correctly on final attempt.
    # But as a safeguard:
    logger.error(
        "Callback to %s ultimately failed after all retries (should have re-raised).",
        callback_url,
    )
    # Consider raising a generic error if not already done.

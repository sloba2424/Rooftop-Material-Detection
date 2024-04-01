def get_utc_timestamp_formatted(time_format: str = "%Y%m%d_%H%M%S") -> str:
    from datetime import datetime

    return datetime.utcnow().strftime(time_format)
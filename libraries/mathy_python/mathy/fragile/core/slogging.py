# This file is copied from https://github.com/src-d/modelforge/blob/master/modelforge/slogging.py
# modelforge is distributed under the Apache License 2.0
import argparse
import codecs
import datetime
import functools
import io
import json
import logging
import os
import re
import sys
import threading
import traceback
from typing import Callable, Dict, Sequence, Tuple, Union

import numpy
import xxhash
import yaml


logs_are_structured = False


def get_datetime_now() -> datetime:
    """
    Return the current UTC date and time.
    """
    return datetime.datetime.now(datetime.timezone.utc)


def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation (for source{d})."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr


timezone, tzstr = get_timezone()
_now = get_datetime_now()
if _now.month == 12:
    _fest = "🎅"
elif _now.month == 10 and _now.day > (31 - 7):
    _fest = "🎃"
else:
    _fest = ""
del _now


def format_datetime(dt: datetime.datetime):
    """Represent the date and time in source{d} format."""
    return dt.strftime("%Y-%m-%dT%k:%M:%S.%f000") + tzstr


def reduce_thread_id(thread_id: int) -> str:
    """Make a shorter thread identifier by hashing the original."""
    return xxhash.xxh32(thread_id.to_bytes(8, "little")).hexdigest()[:4]


def with_logger(cls):
    """Add a logger as static attribute to a class."""
    cls._log = logging.getLogger(cls.__name__)
    return cls


trailing_dot_exceptions = set()


def check_trailing_dot(func: Callable) -> Callable:
    """
    Decorate a function to check if the log message ends with a dot.

    AssertionError is raised if so.
    """

    @functools.wraps(func)
    def decorated_with_check_trailing_dot(record: logging.LogRecord):
        if record.name not in trailing_dot_exceptions:
            msg = record.msg
            if isinstance(msg, str) and msg.endswith(".") and not msg.endswith(".."):
                raise AssertionError(
                    'Log message is not allowed to have a trailing dot: %s: "%s"'
                    % (record.name, msg)
                )
        return func(record)

    return decorated_with_check_trailing_dot


class NumpyLogRecord(logging.LogRecord):
    """
    LogRecord with the special handling of numpy arrays which shortens the long ones.
    """

    @staticmethod
    def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

    def getMessage(self):
        """
        Return the message for this LogRecord.

        Return the message for this LogRecord after merging any user-supplied \
        arguments with the message.
        """
        if isinstance(self.msg, numpy.ndarray):
            msg = self.array2string(self.msg)
        else:
            msg = str(self.msg)
        if self.args:
            a2s = self.array2string
            if isinstance(self.args, Dict):
                args = {
                    k: (a2s(v) if isinstance(v, numpy.ndarray) else v)
                    for (k, v) in self.args.items()
                }
            elif isinstance(self.args, Sequence):
                args = tuple((a2s(a) if isinstance(a, numpy.ndarray) else a) for a in self.args)
            else:
                raise TypeError(
                    "Unexpected input '%s' with type '%s'" % (self.args, type(self.args))
                )
            msg = msg % args
        return msg


class AwesomeFormatter(logging.Formatter):
    """
    logging.Formatter which adds colors to messages and shortens thread ids.
    """

    GREEN_MARKERS = [
        " ok",
        "ok:",
        "finished",
        "complete",
        "ready",
        "done",
        "running",
        "success",
        "saved",
    ]
    GREEN_RE = re.compile("|".join(GREEN_MARKERS))

    def formatMessage(self, record: logging.LogRecord) -> str:
        """Convert the already filled log record to a string."""
        level_color = "0"
        text_color = "0"
        fmt = ""
        if record.levelno <= logging.DEBUG:
            fmt = "\033[0;37m" + logging.BASIC_FORMAT + "\033[0m"
        elif record.levelno <= logging.INFO:
            level_color = "1;36"
            lmsg = record.message.lower()
            if self.GREEN_RE.search(lmsg):
                text_color = "1;32"
        elif record.levelno <= logging.WARNING:
            level_color = "1;33"
        elif record.levelno <= logging.CRITICAL:
            level_color = "1;31"
        if not fmt:
            fmt = (
                "\033["
                + level_color
                + "m%(levelname)s\033[0m:%(rthread)s:%(name)s:\033["
                + text_color
                + "m%(message)s\033[0m"
            )
        fmt = _fest + fmt
        record.rthread = reduce_thread_id(record.thread)
        return fmt % record.__dict__


class StructuredHandler(logging.Handler):
    """logging handler for structured logging."""

    def __init__(self, level=logging.NOTSET):
        """Initialize a new StructuredHandler."""
        super().__init__(level)
        self.local = threading.local()

    @check_trailing_dot
    def emit(self, record: logging.LogRecord):
        """Print the log record formatted as JSON to stdout."""
        created = datetime.datetime.fromtimestamp(record.created, timezone)
        obj = {
            "level": record.levelname.lower(),
            "msg": record.msg % record.args,
            "source": "%s:%d" % (record.filename, record.lineno),
            "time": format_datetime(created),
            "thread": reduce_thread_id(record.thread),
        }
        if record.exc_info is not None:
            obj["error"] = traceback.format_exception(*record.exc_info)[1:]
        try:
            obj["context"] = self.local.context
        except AttributeError:
            pass
        json.dump(obj, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def flush(self):
        """Write all pending text to stdout."""
        sys.stdout.flush()


def setup(level: Union[str, int], structured: bool, config_path: str = None):
    """
    Make stdout and stderr unicode friendly in case of misconfigured \
    environments, initializes the logging, structured logging and \
    enables colored logs if it is appropriate.

    Args:
        level: The global logging level.
        structured: Output JSON logs to stdout.
        config_path: Path to a yaml file that configures the level of output of the loggers. \
                        Root logger level is set through the level argument and will override any \
                        root configuration found in the conf file.

    Returns:
        None

    """
    global logs_are_structured
    logs_are_structured = structured

    if not isinstance(level, int):
        level = logging._nameToLevel[level]

    def ensure_utf8_stream(stream):
        if not isinstance(stream, io.StringIO) and hasattr(stream, "buffer"):
            stream = codecs.getwriter("utf-8")(stream.buffer)
            stream.encoding = "utf-8"
        return stream

    sys.stdout, sys.stderr = (ensure_utf8_stream(s) for s in (sys.stdout, sys.stderr))

    # basicConfig is only called to make sure there is at least one handler for the root logger.
    # All the output level setting is down right afterwards.
    logging.basicConfig()
    logging.setLogRecordFactory(NumpyLogRecord)
    if config_path is not None and os.path.isfile(config_path):
        with open(config_path) as fh:
            config = yaml.safe_load(fh)
        for key, val in config.items():
            logging.getLogger(key).setLevel(logging._nameToLevel.get(val, level))
    root = logging.getLogger()
    root.setLevel(level)

    if not structured:
        handler = root.handlers[0]
        handler.emit = check_trailing_dot(handler.emit)
        if not hasattr(sys.stdin, "closed"):
            handler.setFormatter(AwesomeFormatter())
        elif not sys.stdin.closed and sys.stdout.isatty():
            handler.setFormatter(AwesomeFormatter())
    else:
        root.handlers[0] = StructuredHandler(level)


def set_context(context):
    """Assign the logging context - an abstract object - to the current thread."""
    try:
        handler = logging.getLogger().handlers[0]
    except IndexError:
        # logging is not initialized
        return
    if not isinstance(handler, StructuredHandler):
        return
    handler.acquire()
    try:
        handler.local.context = context
    finally:
        handler.release()


def add_logging_args(
    parser: argparse.ArgumentParser, patch: bool = True, erase_args: bool = True
) -> None:
    """
    Add command line flags specific to logging.

    Args:
        parser: `argparse` parser where to add new flags.
        erase_args: Automatically remove logging-related flags from parsed args.
        patch: Patch parse_args() to automatically setup logging.

    """
    parser.add_argument(
        "--log-level", default="INFO", choices=logging._nameToLevel, help="Logging verbosity."
    )
    parser.add_argument(
        "--log-structured",
        action="store_true",
        help="Enable structured logging (JSON record per line).",
    )
    parser.add_argument(
        "--log-config", help="Path to the file which sets individual log levels of domains."
    )
    # monkey-patch parse_args()
    # custom actions do not work, unfortunately, because they are not invoked if
    # the corresponding --flags are not specified

    def _patched_parse_args(args=None, namespace=None) -> argparse.Namespace:
        args = parser._original_parse_args(args, namespace)
        setup(args.log_level, args.log_structured, args.log_config)
        if erase_args:
            for log_arg in ("log_level", "log_structured", "log_config"):
                delattr(args, log_arg)
        return args

    if patch and not hasattr(parser, "_original_parse_args"):
        parser._original_parse_args = parser.parse_args
        parser.parse_args = _patched_parse_args

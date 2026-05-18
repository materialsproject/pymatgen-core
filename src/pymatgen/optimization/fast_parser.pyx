# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# distutils: language = c

from libc.stddef cimport ptrdiff_t


DEF BUFFER_SIZE = 1048576


cdef extern from *:
    """
    #define FFC_IMPL
    #include "ffc.h"

    static const ffc_parse_options PMG_FFC_PARSE_OPTIONS = {
        FFC_PRESET_GENERAL |
            FFC_FORMAT_FLAG_ALLOW_LEADING_PLUS |
            FFC_FORMAT_FLAG_SKIP_WHITE_SPACE,
        '.'
    };

    static inline int pmg_ffc_parse_one(
        const char* first,
        const char* last,
        double* value,
        const char** next
    ) {
        ffc_result result = ffc_from_chars_double_options(
            first,
            last,
            value,
            PMG_FFC_PARSE_OPTIONS
        );
        if (result.outcome != FFC_OUTCOME_OK) {
            return 0;
        }
        *next = result.ptr;
        return 1;
    }
    """
    int pmg_ffc_parse_one(
        const char* first,
        const char* last,
        double* value,
        const char** next,
    ) noexcept nogil


cdef const char* parse_doubles(
    const char* start,
    Py_ssize_t length,
    double* out,
    Py_ssize_t max_count,
    Py_ssize_t* parsed,
) noexcept nogil:
    """Parse doubles from a char buffer into ``out`` and return the stop pointer."""
    cdef const char* ptr = start
    cdef const char* end = start + length
    cdef const char* next_ptr
    cdef Py_ssize_t count = 0

    while count < max_count and ptr < end:
        if pmg_ffc_parse_one(ptr, end, &out[count], &next_ptr) == 0:
            break
        ptr = next_ptr
        count += 1

    parsed[0] = count
    return ptr


cdef Py_ssize_t last_whitespace(bytes data):
    """Return the index after the last ASCII whitespace byte, or -1."""
    cdef Py_ssize_t idx = len(data) - 1
    cdef unsigned char value

    while idx >= 0:
        value = data[idx]
        if value in (9, 10, 11, 12, 13, 32):
            return idx + 1
        idx -= 1

    return -1


cdef bint only_whitespace_left(bytes data, Py_ssize_t start, Py_ssize_t end):
    """Return whether ``data[start:end]`` contains only ASCII whitespace."""
    cdef Py_ssize_t idx = start
    cdef unsigned char value

    while idx < end:
        value = data[idx]
        if value not in (9, 10, 11, 12, 13, 32):
            return False
        idx += 1

    return True


def parse_N_doubles(file, double[::1] out, Py_ssize_t nelem=-1):
    """Parse doubles from a binary file object into ``out``.

    The file is read in 1 MiB chunks. On return, the file position follows the
    stop pointer returned by the parser without rewinding trailing whitespace.

    Returns:
        int: Parsed element count.
    """
    cdef Py_ssize_t total_parsed = 0
    cdef Py_ssize_t parsed = 0
    cdef Py_ssize_t limit = out.shape[0] if nelem < 0 else nelem
    cdef Py_ssize_t remaining = limit
    cdef Py_ssize_t parse_len = 0
    cdef Py_ssize_t stop_offset
    cdef Py_ssize_t parsed_offset = 0
    cdef Py_ssize_t base_pos = file.tell()
    cdef Py_ssize_t consumed = 0
    cdef const char* start
    cdef const char* stop
    cdef bytes chunk
    cdef bytes buffer = b""
    cdef bint eof = False

    if limit > out.shape[0]:
        raise ValueError("nelem exceeds output length")

    while remaining > 0:
        chunk = file.read(BUFFER_SIZE)
        if not isinstance(chunk, bytes):
            raise TypeError("parse_N_doubles requires a binary file object")

        if chunk:
            buffer += chunk
        else:
            eof = True

        if not buffer:
            break

        parse_len = len(buffer) if eof else last_whitespace(buffer)
        if parse_len < 0:
            if eof:
                parse_len = len(buffer)
            else:
                continue

        if parse_len == 0 and not eof:
            continue

        start = buffer
        stop = parse_doubles(start, parse_len, &out[total_parsed], remaining, &parsed)
        stop_offset = <ptrdiff_t>(stop - start)
        parsed_offset = stop_offset
        total_parsed += parsed
        remaining -= parsed

        if parsed == 0 or remaining == 0:
            file.seek(base_pos + consumed + stop_offset)
            return total_parsed

        if stop_offset < parse_len:
            if not only_whitespace_left(buffer, stop_offset, parse_len):
                file.seek(base_pos + consumed + stop_offset)
                return total_parsed

        consumed += parse_len
        buffer = buffer[parse_len:]

        if eof:
            break

    file.seek(base_pos + consumed + parsed_offset - parse_len)
    return total_parsed

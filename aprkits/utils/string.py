import inspect
import multiprocessing as mp
import re
import threading


def _ncalls(frameinfo):
    with mp.Lock():
        native_id = threading.get_native_id()
        prevframeinfo = __manager_dict.get(native_id)
        if prevframeinfo is not None:
            prevframeinfo = prevframeinfo.get('prevframe')
        prevcontext = __manager_dict.get(native_id)
        if prevcontext is not None:
            prevcontext = prevcontext.get('context')

        if id(prevframeinfo) != id(frameinfo.frame) or prevcontext != frameinfo.code_context[0]:
            if native_id in __manager_dict:
                del __manager_dict[native_id]
            __manager_dict[native_id] = {}
            __manager_dict[native_id]['prevframe'] = frameinfo.frame
            __manager_dict[native_id]['context'] = frameinfo.code_context[0]
            __manager_dict[native_id]['ncall'] = 0

        n = __manager_dict[native_id]['ncall']
        __manager_dict[native_id]['ncall'] += 1

        return n


def to_snake(s):
    s1 = __regsnake1.sub(r'\1_\2', s)
    return __regsnake2.sub(r'\1_\2', s1).lower()


def to_camel(s):
    return __regcamel.sub(lambda x: x.group(1).upper(), s)


def nameof(__var):
    frame_info = inspect.stack(1)
    assert len(frame_info) >= 2, f'Callee frame of {nameof.__name__} cannot be {None}.'
    nameof_frame_info = frame_info[1]
    nameof_context = nameof_frame_info.code_context[0]
    named_elements = __regnameof.findall(nameof_context)
    el_idx = _ncalls(nameof_frame_info)
    name = named_elements[el_idx]
    return name


__manager_dict = {}
__regsnake1 = re.compile(r'(.)([A-Z][a-z]+)')
__regsnake2 = re.compile(r'([a-z0-9])([A-Z])')
__regcamel = re.compile(r'[_]+(\S)')
__regnameof = re.compile(rf'(?<={re.escape(nameof.__name__)}\()\S+?(?=\))')

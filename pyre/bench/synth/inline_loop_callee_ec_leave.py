# A caller loop inlines a callee that stops at its own loop header, so the
# rest of the callee runs through CALL_ASSEMBLER. The callee's `ec.leave`
# must still run when a guard between that call and its recorded position
# leaves the trace; otherwise `topframeref` keeps the callee and the frame
# chain loops. The sift helpers are heapq's pure-Python bodies.


def siftdown(heap, startpos, pos):
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = newitem
    siftdown(heap, startpos, pos)


def heappush(heap, item):
    heap.append(item)
    siftdown(heap, 0, len(heap) - 1)


def heappop(heap):
    lastelt = heap.pop()
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        siftup(heap, 0)
        return returnitem
    return lastelt


def heapify(x):
    n = len(x)
    for i in reversed(range(n // 2)):
        siftup(x, i)


seed = 12345


def nxt():
    global seed
    seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    return seed


def check_invariant(heap):
    for pos, item in enumerate(heap):
        if pos:
            parentpos = (pos - 1) >> 1
            assert heap[parentpos] <= item


def run_heapify():
    for size in list(range(30)) + [20000]:
        heap = [nxt() / 2147483648.0 for dummy in range(size)]
        heapify(heap)
        check_invariant(heap)


def run_heapsort():
    sorted_ok = 0
    for trial in range(100):
        size = nxt() % 50
        data = [nxt() % 25 for i in range(size)]
        if trial & 1:
            heap = data[:]
            heapify(heap)
        else:
            heap = []
            for item in data:
                heappush(heap, item)
        heap_sorted = [heappop(heap) for i in range(size)]
        sorted_ok += heap_sorted == sorted(data)
    return sorted_ok


run_heapify()
print("sorted_ok =", run_heapsort())

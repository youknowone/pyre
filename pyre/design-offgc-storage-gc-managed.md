# off-GC storage 정통화 — GC-manage `malloc_raw` indirect storage

## 문제 (NEW-DEVIATION)

pyre의 컨테이너 계열 `W_*Object`는 가변/가변길이 페이로드를 struct 안에 인라인하지
못하므로 (JIT raw-pointer 모델이 `Copy`-friendly 고정크기 struct를 요구), 페이로드를
**간접 포인터** 뒤에 둔다. 현재 그 간접 저장소는 전부 `lltype::malloc_raw`
(`Box::into_raw`, **off-GC**, `lltype.rs:268`)로 할당되고, GC sweep 시 per-type
destructor가 `Box::from_raw`로 **수동 해제**한다 (`eval.rs` `set_object_destructor`
:496, `dict_object_destructor` :467, `bytes_object_destructor` :484, …).

RPython/PyPy에서 이 저장소들은 **전부 GC-managed 참조**다:

| pyre 필드 (`malloc_raw`) | RPython 대응물 | GC 종류 |
|---|---|---|
| `W_SetObject.items: *mut IndexMap` | `W_BaseSetObject.sstorage` = `r_dict` | `GcStruct("dicttable")` + `GcArray(DICTENTRY)` (rdict.py:179/210) |
| `W_DictObject.dstorage: *mut u8` | `W_DictMultiObject.dstorage` = `r_dict` | 동일 |
| `W_ModuleDictObject.dstorage/mstrategy/object_storage` | `ModuleDictStrategy` 저장소들 | GC dict/list |
| `W_BytesObject.data: *mut Vec<u8>` | `W_BytesObject._value: str` | GcStruct str (bytesobject.py:428) |
| `W_BytearrayObject.data` | `W_BytearrayObject._data` = `resizable_list_supporting_raw_ptr` | GC list (bytearrayobject.py:49) |
| `W_UnicodeObject.value: *mut Wtf8Buf` | `W_UnicodeObject._utf8: str` | GcStruct str (unicodeobject.py:59/66) |
| `W_LongObject.num: *mut BigInt` | `W_LongObject.num: rbigint` | GC instance (longobject.py:117/122) |
| `W_TypeObject.name`, `module.name`, `function.name` | 대응 `str` 필드 | GcStruct str |

즉 **모든 `malloc_raw` 간접 저장소는 정통 모델에선 GC edge**다. off-GC는 pyre가
JIT struct를 고정크기로 유지하려고 도입한 편차다. 그 대가:

1. **수동 해제**: per-type destructor에서 `Box::from_raw`. 등록 누락 시 leak,
   aliasing 시 double-free.
2. **필드 재대입이 JIT wall**: PyPy `w_set.sstorage = w_other.get_storage_copy()`
   (setobject.py:875) 같은 **저장소 포인터 재대입**은 정통적으로 `setfield_gc` 한 개.
   pyre에선 재대입하면 옛 off-GC Box를 수동 free해야 하고 (`Box::from_raw`) 그게
   census phaseA wall (`no analyser registered for boxed.Box.from_raw`). 이를 피하려
   `w_set_copy_storage_from`은 `*d.items = clone()`로 **pointee 통째 덮어쓰기** —
   RPython에 없는 `__deref_write` marker (mir.rs:3393)로 lowering된다. 두 경로 모두
   편차다.

이 wall은 gh#346 Slice C의 두 vec!-graph 중 하나(`set_method_difference`의
`set_copy_real`→`w_set_copy_storage_from` chain)를 막고 있고, 더 넓게는 census의
`Box::from_raw`/`__deref_write` 계열 phaseA fail의 공통 뿌리다.

## 목표

간접 저장소 포인터를 **GC edge**로 만든다. 그러면:
- **필드 재대입 = `setfield_gc`** (정통). `sstorage = copy` / `dstorage = storage` /
  `_data = data` 가 IR-level FieldWrite로 lowering → `Box::from_raw`도 `__deref_write`도
  없음.
- **수동 해제 소멸**. 옛 저장소는 GC sweep이 회수. per-type `Box::from_raw` destructor
  전부 제거.
- census phaseA wall 감소 (Slice C의 set-storage-copy wall 포함).

**비목표 (스코프 밖)**: IndexMap/Vec/Wtf8Buf/BigInt 내부 힙 버퍼를 `GcArray`로 재작성
하지 않는다. pyre는 모든 컨테이너에 네이티브 Rust 자료구조를 쓰는 게 확립된 선택이고,
그건 RPython의 `_ll_malloc_entries` 저수준 재현이 아니라 host-container 위임(정당한
PRE-EXISTING-ADAPTATION)이다. 이 epic이 옮기는 건 **저장소 컨테이너를 담는 박스
포인터**를 off-GC Box에서 GC-managed 블록으로 바꾸는 것뿐이다 — "포인터를 GC edge로",
"컨테이너 자체는 그대로 host Rust".

## 선례 — mapdict storage 블록 (이미 있는 패턴)

`W_ObjectObject.storage` (인스턴스 속성값 블록)는 **이미 GC-managed leaf 블록**이다:
- `W_MAPDICT_STORAGE_GC_TYPE_ID = 55` (object_array.rs:40), `try_gc_alloc_stable_raw`로
  non-moving old-gen 할당 (object_array.rs:252)
- `eval.rs:2146` `TypeInfo::varsize(...)` 로 등록 (leaf, 내부 PyObjectRef를 varsize
  walker가 forward)
- instance custom trace가 `storage` 필드 슬롯을 forward해 major GC가 블록을 greying
  (eval.rs:541-545)

이 epic은 같은 패턴을 **host Rust 컨테이너를 담는 GC 블록**으로 일반화한다. 차이:
mapdict 블록은 내부가 `PyObjectRef` 배열(varsize walker가 훑음)이지만, set/dict 저장소
블록은 내부가 `IndexMap`(host 컨테이너, GC가 못 훑음) → **블록에 custom trace + drop
glue destructor**가 필요하다. 이미 `W_SetObject` 자신이 `set_object_custom_trace`로
IndexMap을 훑고 있으므로, 그 trace를 "set → items 블록 → IndexMap" 2단으로 나누기만
하면 된다.

## 접근 A — 저장소 컨테이너를 GC-managed 박스에 담기

핵심 아이디어: `malloc_raw(container)` → `gc_alloc_managed_box(container, TID)`.
GC 블록은 `[GcHeader | container]` 레이아웃(= 현행 `malloc_typed`의 헤더-프리펜드와
동형)이고, 다음 두 훅을 블록 TID에 등록한다:
- **custom trace**: 컨테이너를 훑어 내부 `PyObjectRef` 슬롯을 forward
  (set/dict/moduledict/bytearray-of-objects). bytes/str/long은 내부에 PyObjectRef가
  없으므로 leaf(no trace).
- **destructor**: `drop_in_place::<Container>()` (host 컨테이너의 힙 버퍼 회수).
  옛 `Box::from_raw` 수동 해제가 하던 일을 GC sweep이 대신 호출.

그러면 소유 `W_*Object`의 저장소 필드는 **평범한 GC 포인터 필드**가 되고
(`gc_ptr_offsets`에 등록하거나, 컨테이너가 host라 leaf가 아니면 소유 객체 custom
trace가 블록 슬롯을 forward), **필드 재대입 = `setfield_gc`**.

### 왜 정통인가

RPython rtyper는 모든 `GcStruct`/`GcArray` malloc을 **동형으로** lowering한다
(`malloc` → `gct_fv_gc_malloc` bracket, framework.py:853). pyre의 GC 블록 = 그
`GcStruct`의 Rust 재현이고, 저장소 필드 재대입이 `setfield_gc`가 되는 것은 rdict.py의
`d.entries = newitems` (GcArray 필드 재대입 = setfield_gc)와 정확히 대응한다. 옛
저장소를 GC가 회수하는 것도 RPython 정통(수동 free 없음). custom trace는 pyre가
host 컨테이너를 쓰기 때문에 필요한 PRE-EXISTING-ADAPTATION의 확장이지 새 편차가
아니다 (이미 dict/set/tuple/list가 custom trace를 쓴다).

## 슬라이스 (스코프: 전체 malloc_raw off-GC)

각 슬라이스는 독립적으로 landable(3-backend green + census 비회귀)이어야 한다.
저장소 종류별로 나눈다. 순서는 leverage(Slice C wall) + 리스크로 정렬.

### S0 — GC-managed storage-box 프리미티브 (기반)
`pyre-object/src/lltype.rs` (또는 새 `gc_storage.rs`)에:
- `gc_alloc_storage_box<T>(value: T, tid: u32) -> *mut T` — `try_gc_alloc_stable_raw`
  로 `[header | T]` 할당(non-moving; 저장소는 self-mutating 메서드가 raw self를
  재유도하므로 stable 필요, `malloc_typed_stable` 논리와 동일). 훅 없으면
  `malloc_raw` fallback (단위 테스트).
- 블록 destructor glue: `unsafe fn drop_storage_box::<T>(addr)` = `drop_in_place`.
- 등록 헬퍼: `register_storage_box_type::<T>(gc, custom_trace?, drop_glue)`.
- 새 GC TID 상수 블록 예약 (set-items, dict-object/int/bytes/…, bytes-data,
  bytearray-data, unicode-value, long-num, name-str).
검증: 단위 테스트(할당→drop glue 호출 확인); census 무변화(아직 소비자 없음);
3-backend green. **dormant 기반**, 안전.

### S1 — set/frozenset `items` (Slice C wall 직접 타깃)
- `w_set_new`/`w_frozenset_new`: `malloc_raw(IndexMap)` → `gc_alloc_storage_box`.
- `set_object_custom_trace`를 2단으로: set이 `items` 블록 슬롯 forward → 블록 custom
  trace가 IndexMap 훑음. (또는 set trace가 그대로 IndexMap 훑되 블록 슬롯도 forward해
  greying — mapdict `storage` 패턴.)
- 블록 destructor = `drop IndexMap` glue. `set_object_destructor`
  (`w_set_dealloc_items`/`Box::from_raw`) **제거**.
- `w_set_copy_storage_from`: `*d.items = clone()` (__deref_write) →
  `d.items = gc_alloc_storage_box((*src.items).clone())` (**setfield_gc**). 옛 블록은
  GC가 회수 → `w_set_dealloc_items` 불필요.
- `w_set_difference_update_from_set`의 storage 교체 경로도 재대입으로.
검증: LLBC 재추출(`LLBC_FORCE_REEXTRACT=1 … pyre-object`) 후 census —
`w_set_copy_storage_from` wall(`__deref_write`) 소멸, set-storage-copy chain wall
소멸; 3-backend bit-exact (set 연산 + `frozenset({1})|{2}` GC-stress).

### S2 — regular dict `dstorage` (Object/Int/Bytes/Unicode 전략)
- `w_dict_new`/`w_dict_new_kwargs`/전략 switch: `malloc_raw` → storage-box.
- 각 전략의 `dealloc_storage` (`Box::from_raw`) 제거; `dict_object_custom_trace`
  (strategy `walk_gc_refs`)를 블록-2단으로.
- `switch_to_object_strategy` 등 저장소 교체 = 재대입.
검증: dict 연산 census wall 감소; GC-stress dict.

### S3 — moduledict (dstorage/mstrategy/object_storage 3-way)
`celldict.rs` `ModuleDictStorage` + `ModuleDictStrategy.caches` (Rc<RefCell>!) —
Rc는 GC 블록에 담기 까다로움. 별도 리스크; caches의 GC 관계 재검토 필요.

### S4 — bytes/bytearray `data`
`W_BytesObject.data`/`W_BytearrayObject.data` (`Vec<u8>`, leaf — 내부 PyObjectRef
없음). custom trace 불필요, drop glue만. `bytes_object_destructor`/
`bytearray_object_destructor` 제거.

### S5 — unicode `value`, long `num`, name strings
`Wtf8Buf`/`BigInt`/`String` (전부 leaf). long은 `try_gc_charge_oldgen_external`로
이미 GC에 external byte를 신고 중 — 그 경로와 통합. name string은
typeobject/module/function/typedef.

### S6 — 정리
남은 off-GC `malloc_raw` 감사; `w_*_dealloc_*` / `Box::from_raw` destructor 전멸
확인; `malloc_raw`는 순수 `flavor='raw'` 스크래치(있다면)만 남김. MEMORY 갱신.

## 리스크 / 오픈 이슈

1. **GC 블록 안의 host 컨테이너 drop glue**: GC sweep이 임의 타입 `T::drop`를 호출해야
   함. **인프라 확인됨**: `DestructorFn = unsafe fn(obj_addr: usize)` (trace.rs:344),
   `TypeInfo::with_destructor_fn` 빌더 (trace.rs:483)로 임의 TID에 destructor 부착
   가능; sweep이 `run_destructor` (collector.rs:936)로 호출하고 old-gen death 경로
   (`deal_with_old_objects_with_destructors`, collector.rs:2505)가 실제 실행.
   `TypeInfo::with_destructor` doc는 "foreign `BigInt`"를 명시 선례로 든다
   (trace.rs:451). 남은 건 블록 TID의 destructor를 `drop_in_place::<Container>` glue로
   지정하는 것뿐 — 현행 per-type `Box::from_raw` destructor가 하던 drop을 블록 TID로
   옮기는 것. S0에서 단위 테스트로 확정.
2. **non-moving 요구**: 저장소는 self-mutating 메서드가 raw self 포인터를 재유도하므로
   `try_gc_alloc_stable_raw`(old-gen, non-moving)여야 함 — mapdict storage와 동일.
   moving nursery에 두면 안 됨.
3. **Rc<RefCell> (moduledict caches)**: GC 블록에 non-`'static`/non-Copy Rc를 담는
   soundness. S3에서 별도 판단, 최악의 경우 moduledict는 off-GC 유지(정당한 예외로
   문서화).
4. **`__deref_write`의 다른 소비자**: `w_set_copy_storage_from` 외에 pointee-whole-
   assign을 쓰는 사이트가 더 있는지 census로 확인 — 전부 재대입 형태로 정통화 가능한지.
5. **census 회귀**: 각 슬라이스마다 LLBC 재추출 필수(pyre-object 소스 변경은 census에
   invisible until re-extract). 새 wall(예: 블록 TID 관련 lowering) 안 생기는지 확인.

## 검증 프로토콜 (슬라이스마다)

1. `cargo test -p pyre-object -p pyre-jit` + `cargo check --workspace`.
2. `LLBC_FORCE_REEXTRACT=1 python3 scripts/extract-llbc.py pyre-object` 후 census
   set-diff: 타깃 wall(`__deref_write`/`Box::from_raw`) 소멸, 신규 wall 0, head count
   비증가.
3. `python3 ./pyre/check.py` 3-backend(dynasm/cranelift/wasm) bit-exact.
4. GC-stress 오라클: 해당 컨테이너를 young 원소로 채운 뒤 minor+major GC 유발
   (set/dict/bytearray append loop), leak/UAF/double-free 없음.
5. 회귀 시 슬라이스 단위 revert.

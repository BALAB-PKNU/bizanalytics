"""balab — 비즈니스 애널리틱스 강의 데이터 로더

강의 노트북에서 데이터 확보 배관(다운로드·캐시·미러)을 감추기 위한 헬퍼다.
학생은 노트북에서 다음 두 줄만 본다.

    from balab import load
    retail = load("online_retail_ii")

동작 순서:
  1) 저장소 캐시가 있으면 그대로 읽는다 (강의자·로컬 환경 — 검증 수치 재현).
  2) 없으면(Colab 등) 미러에서 경량 파일을 내려받아 /tmp 에 캐시한다.

미러는 공개 저장소 BALAB-PKNU/bizanalytics-data (GitHub raw)다. 배포 라이선스가
안전한 데이터(UCI CC BY 4.0 등)만 이 방식으로 직접 배포한다.
"""
from pathlib import Path
import datetime as _dt
import hashlib
import json
import os
import shutil
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _download(url, dest):
    """url을 dest로 내려받는다. 타임아웃·재시도, 완료 후 원자적 교체(부분 파일 방지)."""
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (balab-loader)"})
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f)
            part.replace(dest)
            return dest
        except Exception as e:          # 타임아웃·일시 오류 → 재시도
            last = e
    raise last

# ── 강의 공통 차트 스타일 ─────────────────────────────────────────
# 노트북 셀에서 반복되는 치장 코드(facecolor·스파인·틱·그리드·팔레트)를 감춘다.
PALETTE = {
    "line": "#2a78d6", "series2": "#eda100", "ink": "#0b0b0b", "muted": "#898781",
    "grid": "#e1e0d9", "surf": "#fcfcfb", "accent": "#e34948",
    "A": "#cde2fb", "B": "#e7f0fb", "C": "#f5f5f2",
}


def new_ax(figsize=(9, 5.5)):
    """강의 공통 스타일이 적용된 (fig, ax)를 만든다."""
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.patch.set_facecolor(PALETTE["surf"])
    ax.set_facecolor(PALETTE["surf"])
    return fig, ax


def finish(ax, title=None, xlabel=None, ylabel=None, grid_axis="both"):
    """축 마감(스파인·틱·그리드·라벨)을 공통 스타일로 적용한다."""
    if title:
        ax.set_title(title, color=PALETTE["ink"], fontsize=12, loc="left", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=PALETTE["ink"])
    if ylabel:
        ax.set_ylabel(ylabel, color=PALETTE["ink"])
    ax.grid(True, axis=grid_axis, color=PALETTE["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=PALETTE["muted"], labelcolor=PALETTE["muted"])

# 강의자 호스팅 베이스 URL (공개 데이터 미러 저장소, scripts/deploy_public.py가 배포)
MIRROR_BASE = "https://raw.githubusercontent.com/BALAB-PKNU/bizanalytics-data/main"

# 데이터셋 레지스트리: 모듈 폴더 · 캐시 파일명 · 읽기 옵션
#   단일 파일: "cache"(파일명) + "read"(read_csv 옵션)
#   다중 파일: "files"({별칭: 파일명}) + "mirror"(원격 베이스 URL) → load()가 dict 반환
_REGISTRY = {
    "online_retail_ii": {
        "module": "week02_abc_pareto",
        "cache": "online_retail_ii.csv.gz",
        "read": dict(dtype={"Invoice": "object", "StockCode": "object"},
                     parse_dates=["InvoiceDate"]),
    },
    "cookie_cats": {
        "module": "week01_04_ab_test",
        "cache": "cookie_cats.csv",
        "read": {},
    },
    "callcenter": {
        "module": "week09_queue",
        "cache": "callcenter_1999.csv.gz",
        "read": dict(parse_dates=["arrival"]),
    },
    "loan_process": {
        "module": "week09_queue",
        "cache": "loan_process_2016.csv.gz",
        "read": dict(parse_dates=["timestamp"]),
    },
    "backblaze_2026": {
        "module": "week12_cbm",
        "cache": "backblaze_2026q1.csv.gz",
        "read": dict(parse_dates=["date"]),
    },
    "backblaze_2020": {
        "module": "week12_cbm",
        "cache": "backblaze_2020q1.csv.gz",
        "read": dict(parse_dates=["date"]),
    },
    "upworthy": {
        "module": "week01_04_ab_test",
        "cache": "upworthy.csv.gz",
        "read": {},
    },
    "bike": {
        "module": "week04_regression",
        "cache": "hour.csv",
        "read": dict(parse_dates=["dteday"]),
    },
    "garment": {
        "module": "week04_regression",
        "cache": "garments_worker_productivity.csv",
        "read": {},
    },
    "telco": {
        "module": "week05_06_classification_cost",
        "cache": "telco.csv",
        "read": {},
    },
    "wholesale": {
        "module": "week05_segmentation",
        "cache": "wholesale_customers.csv",
        "read": {},
    },
    "aps": {
        "module": "week05_06_classification_cost",
        "files": {"train": "aps_failure_training_set.csv",
                  "test": "aps_failure_test_set.csv"},
        "read": dict(skiprows=20, na_values="na"),
        "mirror": "https://raw.githubusercontent.com/BALAB-PKNU/bizanalytics-data/main/aps/",
    },
    "instacart": {
        "module": "hw_data",
        "files": {"orders": "instacart_orders.csv.gz",
                  "order_products": "instacart_order_products.csv.gz",
                  "products": "instacart_products.csv.gz",
                  "aisles": "instacart_aisles.csv.gz",
                  "departments": "instacart_departments.csv.gz"},
        "mirror": "https://raw.githubusercontent.com/BALAB-PKNU/bizanalytics-data/main/",
    },
    "dunnhumby": {
        "module": "hw_data",
        "files": {"transactions": "dunnhumby_transactions.csv.gz",
                  "products": "dunnhumby_products.csv.gz",
                  "demographic": "dunnhumby_demographic.csv"},
        "mirror": "https://raw.githubusercontent.com/BALAB-PKNU/bizanalytics-data/main/",
    },
    "vn1": {
        "module": "hw_data",
        "files": {"sales": "vn1_sales.csv.gz", "price": "vn1_price.csv.gz",
                  "holdout": "vn1_sales_holdout.csv.gz"},
        "mirror": "https://raw.githubusercontent.com/BALAB-PKNU/bizanalytics-data/main/",
    },
    "steel_energy": {
        "module": "hw_data",
        "cache": "steel_energy.csv",
        "read": {},
    },
    "secom": {
        "module": "hw_data",
        "cache": "secom.csv.gz",
        "read": {},
    },
    "hillstrom": {
        "module": "hw_data",
        "cache": "hillstrom.csv",
        "read": {},
    },
    "backblaze": {
        "module": "hw_data",
        "cache": "backblaze_st12000_q1_2024.csv.gz",
        "read": dict(parse_dates=["date"]),
    },
    "metro": {
        "module": "week04_regression",
        "cache": "metro_traffic.csv.gz",
        "read": dict(parse_dates=["date_time"]),
    },
    "hotel": {
        "module": "week07_09_forecast_inventory",
        "cache": "hotel_bookings.csv.gz",
        "read": dict(parse_dates=["arrival_date", "status_date"]),
    },
    "olist": {
        "module": "week03_olist_leadtime",
        "files": {
            "orders": "olist_orders_dataset.csv",
            "items": "olist_order_items_dataset.csv",
            "customers": "olist_customers_dataset.csv",
            "reviews": "olist_order_reviews_dataset.csv",
        },
        "mirror": ("https://huggingface.co/datasets/aviahYadler/"
                   "Olist_Ecommerce_Dataset/resolve/main/"),
    },
}


def _find_repo_root():
    here = Path.cwd().resolve()
    for p in [here] + list(here.parents):
        if (p / "code").is_dir():
            return p
    return None


def _read(path, spec):
    return pd.read_csv(path, **spec["read"])


# ── 특수 형식(pickle·xlsx·json)·대용량 데이터의 로더 ────────────────
# 노트북마다 흩어져 있던 배관(경로 탐색·다운로드·형식별 읽기)을 여기로 모은다.
# 학생 노트북은 load("m5") 한 줄만 본다.
def _get(rel):
    """repo 캐시가 있으면 그 경로를, 없으면 미러에서 /tmp로 받아 경로를 돌려준다."""
    fname = Path(rel).name
    root = _find_repo_root()
    if root is not None and (root / rel).exists():
        return root / rel
    tmp = Path("/tmp") / fname
    if not tmp.exists():
        _download(f"{MIRROR_BASE}/{fname}", tmp)
    return tmp


def _load_citibike():
    info = json.loads(Path(_get(
        "code/week11_12_optimization/data/citibike_station_information.json")).read_text())
    status = json.loads(Path(_get(
        "code/week11_12_optimization/data/citibike_station_status.json")).read_text())
    st = pd.DataFrame(info["data"]["stations"])[["station_id", "name", "lat", "lon", "capacity"]]
    av = pd.DataFrame(status["data"]["stations"])[["station_id", "num_bikes_available"]]
    return st.merge(av, on="station_id", how="inner")


_CUSTOM = {
    "m5": lambda: pd.read_pickle(_get(
        "code/week07_09_forecast_inventory/data/m5_ca1_subset.pkl")),
    "flotation": lambda: pd.read_csv(_get(
        "code/week10_spc_quality/flotation_hourly.csv"),
        parse_dates=["date"], index_col="date"),
    "criteo": lambda: pd.read_pickle(_get(
        "code/week13_uplift/data/criteo_uplift_sample.pkl")),
    "brunel": lambda: pd.read_excel(_get(
        "code/week11_12_optimization/data/supply_chain_logistics_problem.xlsx"),
        sheet_name=None),
    "citibike": _load_citibike,
}


def datadir(name):
    """데이터셋 원본 파일이 준비된 디렉터리 경로를 돌려준다(없으면 내려받는다).

    load()가 DataFrame을 돌려주는 것과 달리, 원본 CSV를 직접 읽는
    노트북(감사 과제 등)이 파일 경로가 필요할 때 쓴다.
    """
    spec = _REGISTRY[name]
    fnames = list(spec["files"].values()) if "files" in spec else [spec["cache"]]
    root = _find_repo_root()
    if root is not None:
        local = root / "code" / spec["module"] / "data"
        if all((local / f).exists() for f in fnames):
            return local
    base = spec.get("mirror", f"{MIRROR_BASE}/")
    for f in fnames:
        tmp = Path("/tmp") / f
        if not tmp.exists():
            _download(base + f, tmp)
    return Path("/tmp")


def load(name):
    """이름으로 데이터셋을 반환한다. 단일 파일은 DataFrame, 다중 파일은 dict."""
    if name in _CUSTOM:
        return _CUSTOM[name]()
    if name not in _REGISTRY:
        raise KeyError(f"알 수 없는 데이터셋: {name}. 사용 가능: {list(_REGISTRY) + list(_CUSTOM)}")
    spec = _REGISTRY[name]
    root = _find_repo_root()

    if "files" in spec:   # 다중 파일 → {별칭: DataFrame}
        read = spec.get("read", {})
        out = {}
        for alias, fname in spec["files"].items():
            local = root / "code" / spec["module"] / "data" / fname if root else None
            if local is not None and local.exists():
                out[alias] = pd.read_csv(local, **read)
            else:
                tmp = Path("/tmp") / fname
                if not tmp.exists():
                    _download(spec["mirror"] + fname, tmp)
                out[alias] = pd.read_csv(tmp, **read)
        return out

    if root is not None:
        local = root / "code" / spec["module"] / "data" / spec["cache"]
        if local.exists():
            return _read(local, spec)
    tmp = Path("/tmp") / spec["cache"]
    if not tmp.exists():
        _download(f"{MIRROR_BASE}/{spec['cache']}", tmp)
    return _read(tmp, spec)


# ── 실습 자가 점검 ───────────────────────────────────────────────
# 정답 값은 노트북에도 이 파일에도 평문으로 두지 않는다. 값을 허용오차 격자로
# 양자화한 뒤 해시만 싣고, 학생 값의 해시가 표와 맞는지만 본다.
#
#   from balab import checker, summary
#   check = checker("week10")                      # 셋업 셀에서 한 번
#   check(1, zero_frac=(zero_frac, 0.001), Y=Y)    # 각 과제의 검증 셀에서
#   summary()                                      # 제출 전 마지막 셀에서
#
# 값은 그대로 주거나 (값, 허용오차) 꼴로 준다. 허용오차는 정답이 아니므로
# 노트북에 남아도 된다. 정수·문자열·불리언·날짜는 정확히 일치해야 한다.
# 표·배열은 크기만, 원소 12개 이하의 나열형은 원소마다 확인한다.
#
# 정답표는 scripts/make_answer_key.py가 정답판 노트북을 실행해서 만든다.
# 환경변수 BALAB_ANSWER_KEY가 있으면 check()는 검사 대신 관측값을 그 파일에
# 적는다 — 정답표 생성과 교수 측 채점(scripts/grade_lab.py)이 같은 경로를 쓴다.

_SALT = "balab-bizanalytics-2026"
_ANSWER_DIGESTS = {}   # "week|task|name" -> [kind, tol, digest]
_RESULTS = {}          # (week, task) -> "통과" | "미작성" | "틀림" | "확인불가"
_WEEK = None

_MAX_ITEMS = 12        # 원소별로 확인하는 나열형의 최대 길이
_REL_TOL = 0.01        # 허용오차를 주지 않은 실수의 기본 상대오차


def _digest(key, token):
    # scrypt(메모리-하드)라 한 번 대조에 수십 ms가 든다. 검증 셀 한 번에는
    # 몇 항목뿐이라 체감이 없지만, 후보 값을 전부 넣어보는 전수대조는
    # 항목당 수 시간~수 일이 되도록 일부러 느리게 잡았다.
    h = hashlib.scrypt(str(token).encode(), salt=("%s|%s" % (_SALT, key)).encode(),
                       n=16384, r=8, p=1, maxmem=64 * 1024 * 1024, dklen=8)
    return h.hex()


def _canon(value, tol):
    """값을 (종류, 실효 허용오차, 허용 토큰들)로 환원한다.

    첫 토큰이 정답표에 실리는 대표값이다. 실수는 허용오차의 절반을 격자로 삼아
    정수 눈금으로 양자화하고, 자기 눈금과 좌우 한 칸까지 맞다고 본다 — 허용 폭이
    대략 원래의 절대오차와 같아진다.
    """
    if isinstance(value, (bool, np.bool_)):
        return "bool", None, [str(bool(value))]
    if isinstance(value, str):
        return "str", None, [value.strip()]
    if isinstance(value, (np.datetime64, pd.Timestamp, _dt.date, _dt.datetime)):
        return "date", None, [str(pd.Timestamp(value).date())]
    if isinstance(value, pd.DataFrame):
        return "rows", None, [str(len(value))]
    if isinstance(value, (set, frozenset, pd.Index)):
        return "size", None, [str(len(value))]
    if isinstance(value, np.ndarray) and value.ndim >= 2:
        return "shape", None, [str(tuple(value.shape))]
    if isinstance(value, (int, np.integer)) and not tol:
        return "int", None, [str(int(value))]
    if isinstance(value, (int, float, np.number)):
        eff = float(tol) if tol else max(abs(float(value)), 1.0) * _REL_TOL
        n = int(round(float(value) / (eff / 2.0)))
        return "num", eff, [str(n), str(n - 1), str(n + 1)]
    raise TypeError("check()가 다루지 않는 값이다: %s" % type(value).__name__)


def _split_tol(given):
    """check()의 인자를 (값, 허용오차)로 가른다.

    뒤가 양수인 두 칸짜리 튜플만 허용오차를 붙인 것으로 본다. 앞은 수치·None
    이거나 나열형(리스트·배열·Series·dict)이고, 나열형이면 오차가 원소마다
    적용된다. 원소가 둘인 나열형을 통째로 확인하려면 튜플이 아니라 리스트로 준다.
    """
    if (isinstance(given, tuple) and len(given) == 2
            and isinstance(given[1], (int, float)) and not isinstance(given[1], bool)
            and given[1] > 0
            and (given[0] is None
                 or isinstance(given[0], (list, np.ndarray, pd.Series, dict))
                 or (isinstance(given[0], (int, float, np.number))
                     and not isinstance(given[0], bool)))):
        return given[0], float(given[1])
    return given, None


def _spread(name, value):
    """나열형(리스트·Series·dict)을 [(하위이름, 원소)]로 편다. 긴 것은 길이만."""
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, pd.Series):
        items = list(value.items())
    elif isinstance(value, (list, tuple)) or (isinstance(value, np.ndarray) and value.ndim == 1):
        items = list(enumerate(list(value)))
    else:
        return [(name, value)]
    if len(items) > _MAX_ITEMS:
        return [("%s#len" % name, len(items))]
    return [("%s#%s" % (name, k), v) for k, v in items]


def _plain(v):
    """정답표에 사람이 읽도록 남기는 값."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 6)
    if isinstance(v, pd.DataFrame):
        return [len(v), len(v.columns)]
    if isinstance(v, (set, frozenset, pd.Index)):
        return len(v)
    if isinstance(v, np.ndarray):
        return list(v.shape)
    return str(v)


def check(task, **values):
    """과제 하나의 결과 변수들을 확인한다. 모두 맞으면 True."""
    if _WEEK is None:
        raise RuntimeError('checker("weekNN")으로 주차를 먼저 정한다.')

    flat, blank = [], []
    for name, given in values.items():
        v, tol = _split_tol(given)
        if v is None:
            blank.append(name)
            continue
        for sub, item in _spread(name, v):
            flat.append((sub, item, tol))

    rec = os.environ.get("BALAB_ANSWER_KEY")
    if rec:                                   # 정답 기록 · 제출 채점용 관측
        rows = [{"week": _WEEK, "task": task, "name": n, "kind": None,
                 "tol": None, "token": None, "value": None} for n in blank]
        for sub, item, tol in flat:
            try:
                kind, eff, tokens = _canon(item, tol)
            except TypeError as e:
                rows.append({"week": _WEEK, "task": task, "name": sub,
                             "kind": "오류", "tol": None, "token": str(e), "value": None})
                continue
            rows.append({"week": _WEEK, "task": task, "name": sub, "kind": kind,
                         "tol": eff, "token": tokens[0], "value": _plain(item)})
        with open(rec, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("기록: 과제 %s (%d개 항목)" % (task, len(rows)))
        return True

    if blank:
        _RESULTS[(_WEEK, task)] = "미작성"
        print("미작성: 과제 %s — %s" % (task, ", ".join(blank)))
        return False

    ok, bad, unknown = [], [], []
    for sub, item, _tol in flat:
        key = "%s|%s|%s" % (_WEEK, task, sub)
        spec = _ANSWER_DIGESTS.get(key)
        if spec is None:
            unknown.append(sub)
            continue
        _kind, tol, want = spec
        if _kind == "int" and isinstance(item, (float, np.floating)) and float(item).is_integer():
            item = int(item)          # 187.0으로 구한 값도 187로 본다
        try:
            _k, _eff, tokens = _canon(item, tol)
        except TypeError:
            bad.append(sub)
            continue
        (ok if any(_digest(key, t) == want for t in tokens) else bad).append(sub)

    if unknown and not ok and not bad:
        _RESULTS[(_WEEK, task)] = "확인불가"
        print("확인 불가: 과제 %s — 정답표에 없는 항목 %s" % (task, ", ".join(unknown)))
        return False
    if bad:
        _RESULTS[(_WEEK, task)] = "틀림"
        print("다시: 과제 %s — %s의 값이 기대와 다르다 (%d/%d 항목 통과)"
              % (task, ", ".join(sorted({s.split("#")[0] for s in bad})),
                 len(ok), len(ok) + len(bad)))
        return False
    _RESULTS[(_WEEK, task)] = "통과"
    print("통과: 과제 %s (%d개 항목 확인)" % (task, len(ok)))
    return True


def checker(week):
    """이 노트북이 어느 주차인지 정하고 check 함수를 돌려준다."""
    global _WEEK
    _WEEK = week
    _RESULTS.clear()
    return check


def summary():
    """제출 전 마지막 셀. 과제별 판정을 한 줄로 요약한다."""
    if _WEEK is None:
        print("확인 불가 — checker()로 주차를 정하지 않았다.")
        return False
    tasks = sorted(_RESULTS, key=lambda k: str(k[1]))
    blank = [str(t) for (w, t) in tasks if _RESULTS[(w, t)] == "미작성"]
    wrong = [str(t) for (w, t) in tasks if _RESULTS[(w, t)] not in ("통과", "미작성")]
    done = [t for t in tasks if _RESULTS[t] == "통과"]
    if not tasks:
        print("미완료 — 미작성 0개: - · 틀림 0개: - (검증 셀을 아직 실행하지 않았다)")
        return False
    if not blank and not wrong:
        print("완료 확인: 전체 통과 (%d개 과제)" % len(done))
        return True
    print("미완료 — 미작성 %d개: %s · 틀림 %d개: %s"
          % (len(blank), ", ".join("과제 " + b for b in blank) or "-",
             len(wrong), ", ".join("과제 " + w for w in wrong) or "-"))
    return False


# ── 정답표 시작 (scripts/make_answer_key.py가 다시 쓴다. 손으로 고치지 않는다) ──
_ANSWER_DIGESTS.update({
    'week01|1|n_rows': ['int', None, '525163089d01317c'],
    'week01|1|n_tests': ['int', None, '40127262aab3f036'],
    'week01|2|arms': ['rows', None, 'e99efc042db94933'],
    'week01|3|n_two': ['int', None, 'ca45a1d306db0af9'],
    'week01|4|ctr_diff': ['num', 0.05, '16117b331a9a4b76'],
    'week01|5|bars': ['bool', None, '4df8ed9463de36cd'],
    'week01|5|title': ['bool', None, 'e4e1db2ddceded55'],
    'week01|5|value_labels': ['bool', None, '680042729ff3409b'],
    'week01|5|ylabel': ['bool', None, '9dcc3fb5809914ab'],
    'week01|6|p_tiny': ['bool', None, 'c76998418d3f2ac5'],
    'week01|6|z_abs': ['num', 0.02, 'c0f3adae881393ba'],
    'week01|7|sig_share': ['num', 0.01, '523ece34397c963d'],
    'week02|1|day_max': ['int', None, 'ec696db6acb65591'],
    'week02|1|span_median': ['num', 0.5, '1dcd9cc55f440d1c'],
    'week02|2|n_lines': ['int', None, '4c3c4f155480b899'],
    'week02|2|n_orders': ['int', None, '0fa9eb8baeb5ad5c'],
    'week02|3|G': ['rows', None, '8408b07747a4eb78'],
    'week02|3|gap_median': ['num', 0.01, 'e66ddccd1f1d23bf'],
    'week02|3|n_pairs': ['int', None, '686e2ac94045926c'],
    'week02|4|Gk': ['rows', None, '90b4103a1b1a884b'],
    'week02|4|gap_med_max': ['num', 0.01, 'fee1f645e1b7a9c9'],
    'week02|4|gap_med_median': ['num', 0.01, '2af24542906879a9'],
    'week02|4|prod': ['rows', None, '521ffbea81d81160'],
    'week02|5|miss_all': ['num', 0.002, '4f054ba555f84e98'],
    'week02|5|miss_max': ['num', 0.005, 'cffc906cfaf66a27'],
    'week02|5|miss_min': ['num', 0.005, '6a49cb66a133390f'],
    'week02|5|n_prod': ['int', None, '3eb6fc1fb89a1265'],
    'week02|5|send_all': ['num', 0.01, 'e192589e738ddb4b'],
    'week02|6|miss_max': ['num', 0.005, 'b4f2d2175d158587'],
    'week02|6|miss_overall': ['num', 0.002, '7e39a4792e220551'],
    'week02|6|n_send': ['int', None, 'ebbe3e5e6a5ed863'],
    'week02|6|send_max': ['num', 0.01, '9278f8eb07d05038'],
    'week02|6|send_min': ['num', 0.01, '95edab2dde96115b'],
    'week02|6|sent_mean': ['num', 0.02, '5c2bfa9872ce83e9'],
    'week03|1|n_drives': ['int', None, 'c6d089fc353abba6'],
    'week03|1|n_failed': ['int', None, 'fc0942a135a45e31'],
    'week03|2|mean_age': ['num', 0.05, 'b9db0513644a0ed8'],
    'week03|3|haz': ['rows', None, '83e0cfcc5314d23c'],
    'week03|4|ratio_197': ['num', 3.0, 'c508411990c75337'],
    'week03|4|signal_tbl': ['rows', None, 'ee95f02cb51b84a9'],
    'week03|5|best_breakeven': ['num', 1.5, '7bea5dc41867ecd0'],
    'week03|5|best_policy': ['str', None, '92b7cad608f67b7a'],
    'week03|6|be_age_filter': ['num', 3.0, 'a5e7c668ad97a42c'],
    'week04|1|n_op': ['int', None, '196ca908fac669ee'],
    'week04|1|n_sub': ['int', None, '9aad4a4b78dca23e'],
    'week04|1|op_mean': ['num', 0.05, '03e6171668b5efd1'],
    'week04|1|op_sd': ['num', 0.05, 'b2f05eb530b7a222'],
    'week04|1|sg': ['rows', None, '5f5b4fb7e125c5e8'],
    'week04|1|sg_cols': ['int', None, 'c6aabde65858b9c1'],
    'week04|1|w#len': ['int', None, '9c62ad67b0f20907'],
    'week04|2|cl': ['num', 0.05, 'ebd3ebfa5f5bb189'],
    'week04|2|false_xr': ['num', 0.01, 'e864e956d7f16ad7'],
    'week04|2|lag1': ['num', 0.01, '316a66e89a1165fd'],
    'week04|2|n_ph1': ['int', None, '122327775b5d647b'],
    'week04|2|rbar': ['num', 0.05, 'cc2e0b9108f4ddbc'],
    'week04|2|ucl_xr': ['num', 0.05, '11e3a532d9545b84'],
    'week04|3|lcl': ['num', 0.1, 'b7b2c5a512f6872e'],
    'week04|3|n_alarm_ph1': ['int', None, 'a97bd5af09e55c8f'],
    'week04|3|s_bm': ['num', 0.05, 'e260305bd3ed6981'],
    'week04|3|ucl': ['num', 0.1, 'e3b1ee51a708fe4e'],
    'week04|4|alarm_rate': ['num', 0.005, '8c6a578edd31c5d0'],
    'week04|4|first_signal': ['str', None, '0d7639b1ce5b9489'],
    'week04|4|n_p2': ['int', None, '9e8700cca6ff9d6f'],
    'week04|4|xr_rate': ['num', 0.01, '93b4d75887484936'],
    'week04|5|cap': ['rows', None, '6ca448278d7d5304'],
    'week04|5|cpk_all': ['num', 0.01, 'c7082e3a3d7ee6d9'],
    'week04|5|cpk_ph1': ['num', 0.01, '867dc64fb98aec4c'],
    'week04|5|over_all': ['num', 0.1, '818dc5cf4e919c4e'],
    'week04|6|n_alarm_days': ['int', None, 'e655d9903b9dc77e'],
    'week04|6|over_on_alarm': ['int', None, '84012016441fdb1f'],
    'week04|6|over_total': ['int', None, 'b2f0780cdbff0ed0'],
    'week05|1|X': ['rows', None, '09d7b568e3172e6d'],
    'week05|1|X_cols': ['int', None, '1d52c0fea8a1a2dc'],
    'week05|1|churn_rate': ['num', 0.001, 'f5445783117711d5'],
    'week05|1|n_blank': ['int', None, '2e1eec0326b5f9ec'],
    'week05|2|auc_va': ['num', 0.02, 'e22b688ff7e92f15'],
    'week05|2|n_te': ['int', None, '915646e83c9d0f4c'],
    'week05|2|n_tr': ['int', None, '675253d6316d0e48'],
    'week05|2|n_va': ['int', None, 'f882f83b95b5e097'],
    'week05|3|cost_fn': ['num', 1.0, '21e88c28f17b0fcb'],
    'week05|3|cost_fp': ['num', 0.1, '8b355e38a75cb6dd'],
    'week05|3|loss_months': ['num', 0.05, '3f04b00ac7c6b555'],
    'week05|3|m_churn': ['num', 0.05, '45cf29dc3cb49645'],
    'week05|3|m_stay': ['num', 0.05, 'f48d457ba20f812f'],
    'week05|3|t_theory': ['num', 0.002, '99266d0a8bfed293'],
    'week05|4|cost_val_min': ['num', 2000.0, 'd39eff9339b5df0e'],
    'week05|4|t_star': ['num', 0.02, 'b7abc2dd610d1cd4'],
    'week05|5|cost_all': ['num', 2000.0, '395682c525f43e3e'],
    'week05|5|cost_half': ['num', 2000.0, '35024e8778033a74'],
    'week05|5|cost_none': ['num', 2000.0, 'b8234ba8485dc257'],
    'week05|5|cost_star': ['num', 4000.0, '0921f4f68b67a890'],
    'week05|5|offer_rate': ['num', 0.03, '26fa50f7cf52299a'],
    'week05|6|sens': ['rows', None, '9cb0bde72ca3709e'],
    'week05|6|t_range#0': ['num', 0.03, '0a689fd07e28d357'],
    'week05|6|t_range#1': ['num', 0.03, '046155c6514879a1'],
    'week05|6|t_range#2': ['num', 0.03, '2bc4cc0055253ddb'],
    'week05|6|t_range#3': ['num', 0.03, '8accd752a0c2ae14'],
    'week05|6|t_range#4': ['num', 0.03, '02577157230f327a'],
    'week06|1|n_holiday_days': ['int', None, '4313fc7224945368'],
    'week06|1|n_holiday_hours': ['int', None, '4f304d6d518657ea'],
    'week06|2|mean_holiday': ['num', 1.0, '067fbf638ffa2e83'],
    'week06|2|naive_diff': ['num', 1.0, 'a763673bf312e12b'],
    'week06|3|wd_holiday': ['num', 0.001, '6d203bbf13496c94'],
    'week06|3|wd_normal': ['num', 0.005, 'ee1aaecd30454889'],
    'week06|4|beta_ctrl': ['num', 1.0, '5e7bfe1e65b78ec5'],
    'week06|4|beta_naive': ['num', 1.0, '927230c9d59523db'],
    'week06|5|ci_high': ['num', 2.0, 'b3090d500a3a26d0'],
    'week06|5|ci_low': ['num', 2.0, '98735bee8be3c52f'],
    'week06|5|significant': ['bool', None, '0b5c58a3544aa609'],
    'week06|6|daily_conservative': ['num', 50.0, '0d61affe835b570e'],
    'week06|6|daily_reduction': ['num', 50.0, '73f948af6a91ad2a'],
    'week06|6|pct_reduction': ['num', 0.2, 'b5bda0eb771c4b20'],
    'week07|1|lead#len': ['int', None, 'da9ff277a589a877'],
    'week07|1|mean_lead': ['num', 0.3, '36b7a067e008bf42'],
    'week07|1|n_cases': ['int', None, '010567a7cf48befa'],
    'week07|2|n_w': ['int', None, 'f0a601f76a9bb75a'],
    'week07|2|top_act': ['str', None, '89ba62c20eda2af5'],
    'week07|3|c_agents': ['int', None, 'ca9f21a55adf9811'],
    'week07|3|rho_small': ['bool', None, '792886fb9351531a'],
    'week07|4|required': ['int', None, 'bde9f88a64525838'],
    'week07|5|schedule_share': ['num', 0.01, '56f454551be8b353'],
    'week07|5|suspend_share': ['num', 0.02, 'ff6aaadda86a9ccf'],
    'week07|6|cut_ratio': ['num', 0.01, 'e5a30c5950f9432a'],
    'week07|6|lead_no_queue': ['num', 0.4, '0f2e0d5fb485e5a2'],
    'week09|1|total_deficit': ['int', None, 'aba424f7b2151a63'],
    'week09|1|total_surplus': ['int', None, '2740530a93c6ffd9'],
    'week09|1|zone': ['rows', None, '7f96bebe9f72b935'],
    'week09|2|arc_dist_mean': ['num', 0.02, '1bdfbc6b0898e76e'],
    'week09|2|n_arcs': ['int', None, '632c0a8455a5002a'],
    'week09|2|n_dem': ['int', None, '7c0beb82007aae6c'],
    'week09|2|n_sup': ['int', None, 'a4fb9a956f01124b'],
    'week09|3|base_cost': ['num', 1.0, '247e5bcc6061dcdf'],
    'week09|3|n_moves': ['num', 12.0, 'a614effd767f17f3'],
    'week09|3|status': ['str', None, 'dee9abd501b7efcc'],
    'week09|4|avg_haul': ['num', 0.005, '1bac489ff47ff3ad'],
    'week09|4|n_drained': ['int', None, 'f61f94f10a3f3420'],
    'week09|4|util': ['rows', None, 'e586a52e35f3f0e8'],
    'week09|5|cost_at_1km': ['num', 1.0, 'b7004314938963d6'],
    'week09|5|r_min': ['num', 0.01, '98d67be30ee40f28'],
    'week09|5|radius': ['rows', None, '5d0cd85e957bafa9'],
    'week09|5|radius_status#0': ['str', None, 'dabd10a4d82776e9'],
    'week09|5|radius_status#1': ['str', None, 'd3d367821c17a8d0'],
    'week09|5|radius_status#2': ['str', None, '44c6869ae6eb39ac'],
    'week09|5|radius_status#3': ['str', None, '6c0713656484e28a'],
    'week09|5|radius_status#4': ['str', None, '9db88c2bbf2d7850'],
    'week09|6|cost_03': ['num', 1.0, 'f35d561bc9ebec7e'],
    'week09|6|cost_04': ['num', 1.0, '0fe3db259b7d9c31'],
    'week09|6|fill_status#0': ['str', None, '587ec1f21736cdcf'],
    'week09|6|fill_status#1': ['str', None, '16bcab6fbb348f48'],
    'week09|6|fill_status#2': ['str', None, 'fa112e469086ad30'],
    'week09|6|fill_status#3': ['str', None, '04a2fab5e877f679'],
    'week10|1|Y': ['shape', None, '99ae936b6d908c4a'],
    'week10|1|YH': ['shape', None, '66d73f3935b6993e'],
    'week10|1|zero_frac': ['num', 0.001, '334e563b5c634314'],
    'week10|1|zero_frac_holdout': ['num', 0.001, 'f2ec944dba4e4ef8'],
    'week10|2|n_total': ['int', None, '90bb5c8c7f5da01d'],
    'week10|2|peak_month': ['int', None, 'eab765650fcede54'],
    'week10|2|resid_sd': ['num', 100.0, 'ba314d44023f867f'],
    'week10|2|seas_amp': ['num', 500.0, 'bbc04ec64f7a88ce'],
    'week10|3|interm_share': ['num', 0.002, '3d79fb9dfbac963c'],
    'week10|3|n_intermittent': ['int', None, 'f1dabb42dbeed1fd'],
    'week10|3|n_lumpy': ['int', None, '08c62cb741af70b2'],
    'week10|3|n_smooth': ['int', None, '3c234ce6d694e78d'],
    'week10|4|dead_series': ['int', None, 'd2e3c271d8640708'],
    'week10|4|undef_series': ['int', None, 'b6cbd6930388d83f'],
    'week10|4|zero_cells': ['num', 0.001, 'e34a43ebeea03114'],
    'week10|5|mae': ['rows', None, '0e0345034bb38982'],
    'week10|5|mae_overall#CrostonSBA': ['num', 0.01, 'eafb548d52a3070d'],
    'week10|5|mae_overall#MA13': ['num', 0.01, 'a39f4a297dfa57a3'],
    'week10|5|mae_overall#Naive': ['num', 0.01, '8e8e03a53f3616cb'],
    'week10|5|mae_overall#SES': ['num', 0.01, '21c1774dd739a22c'],
    'week10|5|mae_overall#SNaive52': ['num', 0.01, '6d2df71db3337665'],
    'week10|6|assigned_mean': ['num', 0.01, 'e1347998b8075f4c'],
    'week10|6|best_single': ['str', None, '682f2ccb9b24f5b5'],
    'week10|6|by_class_best#0': ['str', None, 'c3aba92d70b0680e'],
    'week10|6|by_class_best#1': ['str', None, '345f6a26bef96dd4'],
    'week10|6|by_class_best#2': ['str', None, 'e2f800aae4bf2fb2'],
    'week10|6|by_class_best#3': ['str', None, '09fc20b6101ab825'],
    'week10|6|gain': ['num', 0.002, 'affc3614694a9438'],
    'week10|7|share_top200': ['num', 1.0, '3b8ad69709968663'],
    'week10|7|worse_share': ['num', 0.01, '29fc69e59838750d'],
    'week11|1|alive': ['rows', None, '14336f61d18c4b25'],
    'week11|1|noshow_rate': ['num', 0.001, 'c02283ed7e6940f7'],
    'week11|2|HOTEL': ['str', None, 'b4f032d616e711fe'],
    'week11|2|q_rate': ['rows', None, 'f34b81c5430ee3e2'],
    'week11|2|q_rate_cols': ['int', None, '8a2cc2094787326f'],
    'week11|2|res': ['rows', None, 'd69c6a79f018c510'],
    'week11|2|trend_city': ['num', 0.02, '01f40c097cf257e5'],
    'week11|2|trend_resort': ['num', 0.02, 'ecc97785e93deb5b'],
    'week11|3|CAP': ['int', None, '0e700c3b8861fbc2'],
    'week11|3|day': ['rows', None, 'e6318e43f9570bd1'],
    'week11|3|q75': ['num', 1.0, '643d02195b6c2953'],
    'week11|4|n_peak': ['int', None, 'fce925d29500a81d'],
    'week11|4|noshow_mean': ['num', 0.05, '9b1b44c3ea87f1b7'],
    'week11|4|noshow_median': ['num', 0.01, '5d41fb631363858b'],
    'week11|5|ROOM_RATE': ['num', 0.05, '1d0fa4cab97806cf'],
    'week11|5|b_star': ['int', None, '54bb468000f1fb25'],
    'week11|5|cr': ['num', 0.001, '84d88615422bfc65'],
    'week11|5|gain': ['num', 1.0, '853bd2928bfd3238'],
    'week11|6|b_range#0': ['int', None, '2c16c658afeca5ec'],
    'week11|6|b_range#1': ['int', None, '1535b872cdd07ae4'],
    'week11|6|b_range#2': ['int', None, '466f9715d1312c00'],
    'week11|6|b_range#3': ['int', None, 'f910655dac7f3cd5'],
    'week11|6|b_range#4': ['int', None, '66d97416ace9bf4c'],
    'week12|1|balance': ['rows', None, 'aff0b576418c1d00'],
    'week12|1|balance_cols': ['int', None, '0556bfc2a900821f'],
    'week12|1|max_smd': ['num', 0.002, 'c9b3148d61b88ef6'],
    'week12|1|n_base': ['int', None, '54b1bb7d0ddc2500'],
    'week12|1|n_mens': ['int', None, 'b947be9f18c220d6'],
    'week12|1|n_womens': ['int', None, '0ad763e8ccba7f0d'],
    'week12|2|gap_assign': ['num', 0.01, 'eb00312a2ddae2fc'],
    'week12|2|ratio': ['num', 0.1, '07ae737519366bfc'],
    'week12|2|spend_novisit': ['num', 1e-06, 'f13b30b3eff3d02c'],
    'week12|2|spend_visit': ['num', 0.01, 'b126601a651caaa0'],
    'week12|3|ate_cols': ['int', None, '170d9f50b40ef747'],
    'week12|3|ate_spend_mens': ['num', 0.005, 'c55085d7e5eb9d8d'],
    'week12|3|ate_spend_womens': ['num', 0.005, 'bbc447538ebdbd56'],
    'week12|3|ate_table': ['rows', None, 'd77d490275d1aad5'],
    'week12|3|ate_visit_mens': ['num', 0.001, 'b8ae52de0fba3c2e'],
    'week12|3|gap_mw#0': ['num', 0.005, '9b4cbe02e78a7e08'],
    'week12|3|gap_mw#1': ['num', 0.005, '45864df20ba57f19'],
    'week12|3|gap_mw#2': ['num', 0.005, '60e8e031419e4275'],
    'week12|3|lo_spend_mens': ['num', 0.005, '3064ee563390abf8'],
    'week12|4|n_scores': ['int', None, '4f4b1f520a8db199'],
    'week12|4|u_mens_mean': ['num', 0.002, '80941968f40beae2'],
    'week12|4|u_womens_mean': ['num', 0.002, 'ca4711de5a27a727'],
    'week12|4|womens_higher': ['num', 0.02, '556f96aa862bca66'],
    'week12|5|policy_rows#0': ['str', None, '5a83d12226cb79b3'],
    'week12|5|policy_rows#1': ['str', None, '3c3edcda232785b9'],
    'week12|5|policy_rows#2': ['str', None, '75e072b776847688'],
    'week12|5|policy_rows#3': ['str', None, '2c5822397830c9ce'],
    'week12|5|policy_rows#4': ['str', None, '0695debb2f3e8724'],
    'week12|5|policy_rows#5': ['str', None, '41f3fd2370dd5852'],
    'week12|5|spend_uplift_10': ['num', 0.02, 'e660763d31667f5a'],
    'week12|5|visit_uplift#Random 10%': ['num', 0.05, 'c260f24d011d89bf'],
    'week12|5|visit_uplift#Random 20%': ['num', 0.05, '7688c8d1e8f92011'],
    'week12|5|visit_uplift#Random 50%': ['num', 0.05, '08540579f106003d'],
    'week12|5|visit_uplift#Uplift 10%': ['num', 0.05, '57658d9a3166fb80'],
    'week12|5|visit_uplift#Uplift 20%': ['num', 0.05, '4863c181156ce8eb'],
    'week12|5|visit_uplift#Uplift 50%': ['num', 0.05, '9428386540d06909'],
    'week12|6|q_uplift_last': ['num', 1.0, '41411a6574db40dd'],
    'week12|6|qini_coef#Random': ['num', 1.0, '3de60f826845ef97'],
    'week12|6|qini_coef#Uplift': ['num', 1.0, '802f8ba410f2961a'],
    'week12|6|two': ['rows', None, '59c7b17551462822'],
})
# ── 정답표 끝 ───────────────────────────────────────────────────

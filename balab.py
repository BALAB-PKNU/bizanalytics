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
    "loan_process_2012": {  # 같은 은행의 2011-10~2012-03 로그 (BPI Challenge 2012), 활동 이름을 2017 로그에 맞춰 영어로 바꿈
        "module": "week09_queue",
        "cache": "loan_process_2012.csv.gz",
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


def _load_cmp():
    """PHM 2016 CMP 데이터 챌린지 원본 zip을 받아 풀고, 풀린 폴더 경로를 돌려준다.

    폴더 안: training/CMP-training-000.csv … (185개, 연마 중 센서 기록 약 1초에 한 줄),
    CMP-training-removalrate.csv (웨이퍼·단계마다 연마 속도 하나).
    학생 노트북이 파일을 직접 읽어 이어 붙인다.
    """
    import zipfile
    z = _get("code/week10_spc_quality/data/phm2016_cmp_data_set.zip")
    out = Path(z).parent / "phm2016_cmp"
    if not (out / "CMP-training-removalrate.csv").exists():
        (out / "training").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith(".csv") and ("/CMP-data/training/" in n or n.endswith("CMP-training-removalrate.csv")):
                    dest = out / ("training/" + Path(n).name if "/CMP-data/training/" in n else Path(n).name)
                    dest.write_bytes(zf.read(n))
    return out


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
    "cmp": _load_cmp,   # DataFrame이 아니라 원본 파일이 든 폴더 경로를 돌려준다
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
    'week02|1|future_end': ['str', None, '8ad759d31e90b811'],
    'week02|1|future_n': ['int', None, '009178d3dfd658c8'],
    'week02|1|future_start': ['str', None, 'c82466e83148bec3'],
    'week02|1|past_end': ['str', None, 'ce2b3fdb8adf6c3a'],
    'week02|1|past_n': ['int', None, 'e06d09effa3d237a'],
    'week02|1|past_start': ['str', None, '740dfbf7a4cee94f'],
    'week02|1|unavailable': ['int', None, 'bd2eb1451ed301ce'],
    'week02|2|a#AC': ['num', 0.01, 'ea7851b07a0a4cbc'],
    'week02|2|a#AL': ['num', 0.01, '5b6d160fcf9b6df6'],
    'week02|2|a#AM': ['num', 0.01, '101becdc55978a84'],
    'week02|2|a#AP': ['num', 0.01, '10f9b1900af22d8d'],
    'week02|2|a#BA': ['num', 0.01, 'ba545228839aadcc'],
    'week02|2|a#CE': ['num', 0.01, '2d997c393159031d'],
    'week02|2|a#DF': ['num', 0.01, '2b36c848075fc3dc'],
    'week02|2|a#ES': ['num', 0.01, '0c8944788b08a810'],
    'week02|2|a#GO': ['num', 0.01, 'ac1a01004028743f'],
    'week02|2|a#MA': ['num', 0.01, '26e01b3ce91d4794'],
    'week02|2|a#MG': ['num', 0.01, '181ffb88a938eb18'],
    'week02|2|a#MS': ['num', 0.01, '41cd6be5d73f66ab'],
    'week02|2|b#MT': ['num', 0.01, '2073b4312adf883f'],
    'week02|2|b#PA': ['num', 0.01, '1ef89717f771ae68'],
    'week02|2|b#PB': ['num', 0.01, 'a127db364fb6e26b'],
    'week02|2|b#PE': ['num', 0.01, '6c5aa64e6d262149'],
    'week02|2|b#PI': ['num', 0.01, 'cd53aeff6c9788db'],
    'week02|2|b#PR': ['num', 0.01, '48e482924d812faa'],
    'week02|2|b#RJ': ['num', 0.01, 'bfab4c9f4625a435'],
    'week02|2|b#RN': ['num', 0.01, 'd5cb0e73467ba725'],
    'week02|2|b#RO': ['num', 0.01, 'a786b90bc7e3f1f4'],
    'week02|2|b#RR': ['num', 0.01, '72b567b88fcbac51'],
    'week02|2|b#RS': ['num', 0.01, '5e133d88b5706a1e'],
    'week02|2|b#SC': ['num', 0.01, '14d1ae30c89478f4'],
    'week02|2|c#SE': ['num', 0.01, '08a6253e7b2e36c7'],
    'week02|2|c#SP': ['num', 0.01, 'f789fd268aac8425'],
    'week02|2|c#TO': ['num', 0.01, '4b2da95f5f288b65'],
    'week02|2|n': ['int', None, '1e2119682d152fef'],
    'week02|3|a#AC': ['num', 0.01, '87da7e8f923a4b38'],
    'week02|3|a#AL': ['num', 0.01, '98b47d912e3ffb48'],
    'week02|3|a#AM': ['num', 0.01, '06963fcdabf7c269'],
    'week02|3|a#AP': ['num', 0.01, '888a11ca1e6108a7'],
    'week02|3|a#BA': ['num', 0.01, '4ea11e18238410df'],
    'week02|3|a#CE': ['num', 0.01, 'bbf90adb3e5677fc'],
    'week02|3|a#DF': ['num', 0.01, '9711ff09c86ce9a7'],
    'week02|3|a#ES': ['num', 0.01, '9d8e23612903a196'],
    'week02|3|a#GO': ['num', 0.01, '92b3de57f9e3a568'],
    'week02|3|a#MA': ['num', 0.01, 'efb1de528a94cb99'],
    'week02|3|a#MG': ['num', 0.01, 'f6870c6266ac1887'],
    'week02|3|a#MS': ['num', 0.01, '6c7b6ba357348282'],
    'week02|3|b#MT': ['num', 0.01, '2dd9d8b2a17c5fee'],
    'week02|3|b#PA': ['num', 0.01, '1a8c4f9c6f9cb041'],
    'week02|3|b#PB': ['num', 0.01, 'd6e9d87ebb023fc4'],
    'week02|3|b#PE': ['num', 0.01, 'd9a9258f12b1f929'],
    'week02|3|b#PI': ['num', 0.01, '6122b520e6704219'],
    'week02|3|b#PR': ['num', 0.01, '1640bee2be2013ce'],
    'week02|3|b#RJ': ['num', 0.01, 'a50a0e1c1e804029'],
    'week02|3|b#RN': ['num', 0.01, 'dddceba0a8d03974'],
    'week02|3|b#RO': ['num', 0.01, '8c77992ebf1432dc'],
    'week02|3|b#RR': ['num', 0.01, '13edfda604fa3ad9'],
    'week02|3|b#RS': ['num', 0.01, 'bec5b5278e45147e'],
    'week02|3|b#SC': ['num', 0.01, '346ec3a738ce498d'],
    'week02|3|c#SE': ['num', 0.01, '4b2b7ab987f897dd'],
    'week02|3|c#SP': ['num', 0.01, '99204b295ba7f241'],
    'week02|3|c#TO': ['num', 0.01, '4008e766bf847fa0'],
    'week02|3|max_spread': ['num', 0.01, '6e9778aa187a54bd'],
    'week02|3|missing': ['int', None, '164db897fdda2b1b'],
    'week02|3|n': ['int', None, 'bfebdcc1e35a493a'],
    'week02|4|late_n': ['int', None, '985c6fbe945308fe'],
    'week02|4|late_rate': ['num', 0.0001, 'bd785e7b1e79d670'],
    'week02|4|missing': ['int', None, 'b2e763b627189ffa'],
    'week02|5|days#current': ['num', 0.01, '16cc2ac838bf9ea4'],
    'week02|5|days#q92': ['num', 0.01, '70c2fde61c6d6772'],
    'week02|5|n': ['int', None, 'e23d2e56a9ca87cf'],
    'week02|5|rate#current': ['num', 0.0001, '6e1061d272be5e15'],
    'week02|5|rate#q92': ['num', 0.0001, 'b99ab16ae12ab051'],
    'week02|6|late92_a#AC': ['num', 0.0001, '6cca3407c4597a09'],
    'week02|6|late92_a#AL': ['num', 0.0001, '493bdcea85a1390d'],
    'week02|6|late92_a#AM': ['num', 0.0001, 'b194918e21c76b81'],
    'week02|6|late92_a#AP': ['num', 0.0001, '8a9f9c82572931d7'],
    'week02|6|late92_a#BA': ['num', 0.0001, 'c06c59a845fbb932'],
    'week02|6|late92_a#CE': ['num', 0.0001, 'ee3b10bea6243c7b'],
    'week02|6|late92_a#DF': ['num', 0.0001, 'a136bf275c9441dc'],
    'week02|6|late92_a#ES': ['num', 0.0001, '7fa56753e3e863a3'],
    'week02|6|late92_a#GO': ['num', 0.0001, '6df68a47d8ed294d'],
    'week02|6|late92_a#MA': ['num', 0.0001, 'a0d359afa333500d'],
    'week02|6|late92_a#MG': ['num', 0.0001, '3bfa6379bd41ba15'],
    'week02|6|late92_a#MS': ['num', 0.0001, '700004d96f36d57b'],
    'week02|6|late92_b#MT': ['num', 0.0001, '2f28976d8af8afc1'],
    'week02|6|late92_b#PA': ['num', 0.0001, '80154e54328d1ff9'],
    'week02|6|late92_b#PB': ['num', 0.0001, '1e1be65d12205798'],
    'week02|6|late92_b#PE': ['num', 0.0001, '26f8a37570aa5711'],
    'week02|6|late92_b#PI': ['num', 0.0001, '6ef701e0f971156c'],
    'week02|6|late92_b#PR': ['num', 0.0001, '3d946211bd70003c'],
    'week02|6|late92_b#RJ': ['num', 0.0001, '543d8e03811545b9'],
    'week02|6|late92_b#RN': ['num', 0.0001, '31930a906b6ce39e'],
    'week02|6|late92_b#RO': ['num', 0.0001, '8f5e844a1dd7f1fc'],
    'week02|6|late92_b#RR': ['num', 0.0001, 'a3d77e3e722525ad'],
    'week02|6|late92_b#RS': ['num', 0.0001, 'c0c9df13f2d8f6a3'],
    'week02|6|late92_b#SC': ['num', 0.0001, '20ff9944a81415dd'],
    'week02|6|late92_c#SE': ['num', 0.0001, '80cb76ceb3496e9a'],
    'week02|6|late92_c#SP': ['num', 0.0001, '9e47af6fed3b6b82'],
    'week02|6|late92_c#TO': ['num', 0.0001, '9b981d7648de0d92'],
    'week02|6|late_now_a#AC': ['num', 0.0001, '0968ce8f08005685'],
    'week02|6|late_now_a#AL': ['num', 0.0001, 'b9e65f6d50df61cb'],
    'week02|6|late_now_a#AM': ['num', 0.0001, '74ba120e87747f47'],
    'week02|6|late_now_a#AP': ['num', 0.0001, 'efa622ac70d05155'],
    'week02|6|late_now_a#BA': ['num', 0.0001, '06afda2a7ae3cfea'],
    'week02|6|late_now_a#CE': ['num', 0.0001, '2ed8ac61f80a8858'],
    'week02|6|late_now_a#DF': ['num', 0.0001, '2a7aa7bff5a88cac'],
    'week02|6|late_now_a#ES': ['num', 0.0001, '8ddefa01bd2cbf25'],
    'week02|6|late_now_a#GO': ['num', 0.0001, 'f7d0ce85e7084e5f'],
    'week02|6|late_now_a#MA': ['num', 0.0001, '9df926c7553fda1a'],
    'week02|6|late_now_a#MG': ['num', 0.0001, '19a62c5000a3cf58'],
    'week02|6|late_now_a#MS': ['num', 0.0001, '351429fed0b45b2b'],
    'week02|6|late_now_b#MT': ['num', 0.0001, '88337e6c89bc0f50'],
    'week02|6|late_now_b#PA': ['num', 0.0001, '122ffad343227449'],
    'week02|6|late_now_b#PB': ['num', 0.0001, '970d0a684d9ee788'],
    'week02|6|late_now_b#PE': ['num', 0.0001, '18964f4055cd85cf'],
    'week02|6|late_now_b#PI': ['num', 0.0001, '0384fd19687fc902'],
    'week02|6|late_now_b#PR': ['num', 0.0001, 'e8a7e06fec2a1ff5'],
    'week02|6|late_now_b#RJ': ['num', 0.0001, 'ecabbb982ca94b0d'],
    'week02|6|late_now_b#RN': ['num', 0.0001, '1a70fe905be8a497'],
    'week02|6|late_now_b#RO': ['num', 0.0001, 'd5dd30ac3e23e536'],
    'week02|6|late_now_b#RR': ['num', 0.0001, 'e4b8c46f3f242025'],
    'week02|6|late_now_b#RS': ['num', 0.0001, 'ce3f7ba3c3de7a8e'],
    'week02|6|late_now_b#SC': ['num', 0.0001, '49fa9cdef0cefea6'],
    'week02|6|late_now_c#SE': ['num', 0.0001, '5bc0a8ef95017a93'],
    'week02|6|late_now_c#SP': ['num', 0.0001, 'b952bc1e13e248c7'],
    'week02|6|late_now_c#TO': ['num', 0.0001, '80ed87ff3d49d10f'],
    'week02|6|n': ['int', None, 'aed0c7c9980fd7e9'],
    'week02|6|orders_a#AC': ['int', None, '827bbfde5637ac6c'],
    'week02|6|orders_a#AL': ['int', None, '3146877b6139a277'],
    'week02|6|orders_a#AM': ['int', None, '67fe138a5b19abec'],
    'week02|6|orders_a#AP': ['int', None, '13b2b24254df19c1'],
    'week02|6|orders_a#BA': ['int', None, '32536e66c3e1b777'],
    'week02|6|orders_a#CE': ['int', None, 'a6c19c03bc63b67c'],
    'week02|6|orders_a#DF': ['int', None, '112f1326a84d88ea'],
    'week02|6|orders_a#ES': ['int', None, 'b5791c601260d3f5'],
    'week02|6|orders_a#GO': ['int', None, 'e2e8cee9c121a847'],
    'week02|6|orders_a#MA': ['int', None, '9dfab34e5c3d5587'],
    'week02|6|orders_a#MG': ['int', None, 'f5ad752ab9cee48a'],
    'week02|6|orders_a#MS': ['int', None, '5ac97982d5d1a43a'],
    'week02|6|orders_b#MT': ['int', None, 'fc09a5fcb02bd6f2'],
    'week02|6|orders_b#PA': ['int', None, '33fe341f19a69940'],
    'week02|6|orders_b#PB': ['int', None, 'abfc63c98fb6e5a0'],
    'week02|6|orders_b#PE': ['int', None, '7ad2f8f1fc33fad0'],
    'week02|6|orders_b#PI': ['int', None, '5599a374db13a543'],
    'week02|6|orders_b#PR': ['int', None, '268702619c9a6ace'],
    'week02|6|orders_b#RJ': ['int', None, 'c2c4f060da6682bd'],
    'week02|6|orders_b#RN': ['int', None, '78b8b60da189da9e'],
    'week02|6|orders_b#RO': ['int', None, 'eff02bfdaefd83ea'],
    'week02|6|orders_b#RR': ['int', None, '86acb33bf8bb449e'],
    'week02|6|orders_b#RS': ['int', None, '2ce0425172f6fcb5'],
    'week02|6|orders_b#SC': ['int', None, 'e522bed433a4b874'],
    'week02|6|orders_c#SE': ['int', None, 'edb366f36e935144'],
    'week02|6|orders_c#SP': ['int', None, '4e6c42edf4d50154'],
    'week02|6|orders_c#TO': ['int', None, 'ffc50baf371c7728'],
    'week02|6|promise92_a#AC': ['num', 0.01, '9d4a78c403e40100'],
    'week02|6|promise92_a#AL': ['num', 0.01, '16f77d82f82fcae2'],
    'week02|6|promise92_a#AM': ['num', 0.01, '1efc3d2d1269e811'],
    'week02|6|promise92_a#AP': ['num', 0.01, 'c7a38756ac0b288f'],
    'week02|6|promise92_a#BA': ['num', 0.01, 'a780a2759adc0781'],
    'week02|6|promise92_a#CE': ['num', 0.01, '806a44cec9f0d667'],
    'week02|6|promise92_a#DF': ['num', 0.01, 'b9c02a9a2d2b519e'],
    'week02|6|promise92_a#ES': ['num', 0.01, '0db86de7e31ef816'],
    'week02|6|promise92_a#GO': ['num', 0.01, '4de846cc983c83a1'],
    'week02|6|promise92_a#MA': ['num', 0.01, '76f046cac0571077'],
    'week02|6|promise92_a#MG': ['num', 0.01, '1012e48d371eecc9'],
    'week02|6|promise92_a#MS': ['num', 0.01, '9f2d0cffac606345'],
    'week02|6|promise92_b#MT': ['num', 0.01, '94895619cf7ef06a'],
    'week02|6|promise92_b#PA': ['num', 0.01, '7604bb11c68563ed'],
    'week02|6|promise92_b#PB': ['num', 0.01, 'b3b5ddfb616ffbe2'],
    'week02|6|promise92_b#PE': ['num', 0.01, 'c4c63c6008dc41e2'],
    'week02|6|promise92_b#PI': ['num', 0.01, '6a1e27ad7760c7f4'],
    'week02|6|promise92_b#PR': ['num', 0.01, '684aeeccbf2d71d1'],
    'week02|6|promise92_b#RJ': ['num', 0.01, '3319ab3111c71e06'],
    'week02|6|promise92_b#RN': ['num', 0.01, '064328863dbaa78b'],
    'week02|6|promise92_b#RO': ['num', 0.01, 'db2aeb8bd354c683'],
    'week02|6|promise92_b#RR': ['num', 0.01, '2f29951439f08f6b'],
    'week02|6|promise92_b#RS': ['num', 0.01, '2f37c81f75abe3b3'],
    'week02|6|promise92_b#SC': ['num', 0.01, 'dc2a4b33596182a1'],
    'week02|6|promise92_c#SE': ['num', 0.01, '6990610d5efc7d3e'],
    'week02|6|promise92_c#SP': ['num', 0.01, '2f21927b5c26b0eb'],
    'week02|6|promise92_c#TO': ['num', 0.01, '26201d557e8c2666'],
    'week02|6|promise_now_a#AC': ['num', 0.01, '062b4141106ce892'],
    'week02|6|promise_now_a#AL': ['num', 0.01, '6af13e2190a3fa3c'],
    'week02|6|promise_now_a#AM': ['num', 0.01, 'd166d4a8ef4c6341'],
    'week02|6|promise_now_a#AP': ['num', 0.01, '6dcc877f78a6f6f3'],
    'week02|6|promise_now_a#BA': ['num', 0.01, '4e013d97bde49b30'],
    'week02|6|promise_now_a#CE': ['num', 0.01, '5d358fb502415836'],
    'week02|6|promise_now_a#DF': ['num', 0.01, 'c43b8e57aa57b1e1'],
    'week02|6|promise_now_a#ES': ['num', 0.01, '881e126059a43520'],
    'week02|6|promise_now_a#GO': ['num', 0.01, 'e7e6dff4fb5d055d'],
    'week02|6|promise_now_a#MA': ['num', 0.01, '3e60f4112923771b'],
    'week02|6|promise_now_a#MG': ['num', 0.01, '7e98a2c58339e07b'],
    'week02|6|promise_now_a#MS': ['num', 0.01, 'edb86f5fb53c299b'],
    'week02|6|promise_now_b#MT': ['num', 0.01, '03dac760bdebf275'],
    'week02|6|promise_now_b#PA': ['num', 0.01, '11a02aaff3134e34'],
    'week02|6|promise_now_b#PB': ['num', 0.01, 'e619ebf74b07ba11'],
    'week02|6|promise_now_b#PE': ['num', 0.01, 'e979790c656e4144'],
    'week02|6|promise_now_b#PI': ['num', 0.01, '180f1a7776d978f0'],
    'week02|6|promise_now_b#PR': ['num', 0.01, 'dda3440a19b07455'],
    'week02|6|promise_now_b#RJ': ['num', 0.01, '17febb77a9ec72c0'],
    'week02|6|promise_now_b#RN': ['num', 0.01, 'bef6c9937b1fa0fb'],
    'week02|6|promise_now_b#RO': ['num', 0.01, '004a7252beba3b3e'],
    'week02|6|promise_now_b#RR': ['num', 0.01, 'c4a9d22b071e3bc8'],
    'week02|6|promise_now_b#RS': ['num', 0.01, '99072ca7360eef1e'],
    'week02|6|promise_now_b#SC': ['num', 0.01, '92f83632e959b1bc'],
    'week02|6|promise_now_c#SE': ['num', 0.01, '7d1fa2fbb40cb12b'],
    'week02|6|promise_now_c#SP': ['num', 0.01, 'df7a2544ee2a66b2'],
    'week02|6|promise_now_c#TO': ['num', 0.01, '22cf88ce781c2882'],
    'week03|1|before_points': ['int', None, '34d2fa567096f575'],
    'week03|1|end': ['str', None, '37d9adc50f6d798b'],
    'week03|1|first_disks': ['int', None, '337c5ea623c1c04d'],
    'week03|1|first_rows': ['int', None, 'a7ef451d634fe732'],
    'week03|1|n_dates': ['int', None, '42f6d0ff052230ad'],
    'week03|1|n_disks': ['int', None, '45bbcc4f5ed1cd5e'],
    'week03|1|n_failed': ['int', None, 'd580eb86e69160f1'],
    'week03|1|n_points': ['int', None, 'f9d6c1564e8fc62b'],
    'week03|1|start': ['str', None, '8f50abb124c6f096'],
    'week03|2|hit0_size': ['int', None, '17e630e50a22787f'],
    'week03|2|initial0_size': ['int', None, 'bbb11d6aae30c09e'],
    'week03|2|m0': ['int', None, 'e63afa6a00aaa164'],
    'week03|2|n0': ['int', None, 'bd96d4b528dbdeff'],
    'week03|2|pf0': ['num', 0.0001, '3b1ded165bc51831'],
    'week03|3|initial#pending > 0': ['int', None, '3754481ea799344e'],
    'week03|3|initial#pending > 1': ['int', None, '2f7dd36f3429914a'],
    'week03|3|initial#pending > 16': ['int', None, 'db786ed4ae33c9a3'],
    'week03|3|initial#pending > 4': ['int', None, '75d369844a5a392e'],
    'week03|3|n_rules': ['int', None, '88fac76958a88b89'],
    'week03|3|new#pending > 0': ['int', None, '5a3149fd9ebc6c60'],
    'week03|3|new#pending > 1': ['int', None, 'ae9e54851f4325e3'],
    'week03|3|new#pending > 16': ['int', None, 'f7ba392e096dd577'],
    'week03|3|new#pending > 4': ['int', None, 'aac2a71e14de2027'],
    'week03|3|per_failure#pending > 0': ['num', 0.0001, 'c0c11be5024096b9'],
    'week03|3|per_failure#pending > 1': ['num', 0.0001, '80e0bcd1c86041a6'],
    'week03|3|per_failure#pending > 16': ['num', 0.0001, '75e386743e327f4b'],
    'week03|3|per_failure#pending > 4': ['num', 0.0001, 'cc0995d2934e491c'],
    'week03|3|prevented#pending > 0': ['int', None, '7e527488a027b96f'],
    'week03|3|prevented#pending > 1': ['int', None, '67ce292062b90b69'],
    'week03|3|prevented#pending > 16': ['int', None, '731e18a7065f8c03'],
    'week03|3|prevented#pending > 4': ['int', None, '6d62114bdfb55d18'],
    'week03|3|remaining#pending > 0': ['int', None, '25b98475bbd108b0'],
    'week03|3|remaining#pending > 1': ['int', None, '54857b5491093810'],
    'week03|3|remaining#pending > 16': ['int', None, '6c6c5ec4e0356938'],
    'week03|3|remaining#pending > 4': ['int', None, '0777a085d048fc06'],
    'week03|3|replaced#pending > 0': ['int', None, '0a7a3c6bc03ea436'],
    'week03|3|replaced#pending > 1': ['int', None, '75c3315c72ed5429'],
    'week03|3|replaced#pending > 16': ['int', None, 'd685c97f6c0a8a8d'],
    'week03|3|replaced#pending > 4': ['int', None, '7e7a9af263f3beca'],
    'week03|4|best20': ['str', None, 'e624383e5be4a01f'],
    'week03|4|best5': ['str', None, '8ff5ffdfaad811fa'],
    'week03|4|best50': ['str', None, '82efe4cdedd29be3'],
    'week03|4|cost20#after failure': ['num', 0.0001, 'b9549b865fc6f993'],
    'week03|4|cost20#pending > 0': ['num', 0.0001, '9af79039d517ab33'],
    'week03|4|cost20#pending > 1': ['num', 0.0001, '3849524c7a994485'],
    'week03|4|cost20#pending > 16': ['num', 0.0001, 'ea76157af27b300b'],
    'week03|4|cost20#pending > 4': ['num', 0.0001, '95fe17c1270e60b8'],
    'week03|4|cost5#after failure': ['num', 0.0001, '540d1c205d4e457f'],
    'week03|4|cost5#pending > 0': ['num', 0.0001, 'cdedacc68373f1e9'],
    'week03|4|cost5#pending > 1': ['num', 0.0001, 'b5533a75a7ce7247'],
    'week03|4|cost5#pending > 16': ['num', 0.0001, '65dd451a3bc43759'],
    'week03|4|cost5#pending > 4': ['num', 0.0001, '4a7f1b5930020404'],
    'week03|4|cost50#after failure': ['num', 0.0001, '0f9b78e84262be80'],
    'week03|4|cost50#pending > 0': ['num', 0.0001, 'f60f89800bb32951'],
    'week03|4|cost50#pending > 1': ['num', 0.0001, '0857b301da2bb194'],
    'week03|4|cost50#pending > 16': ['num', 0.0001, '26bef68c4048aa05'],
    'week03|4|cost50#pending > 4': ['num', 0.0001, '67d9bf64bd35e78f'],
    'week03|4|n_choices': ['int', None, '097a93a90936eab0'],
    'week04|1|cas_diff': ['num', 0.1, '0395ae721e110c48'],
    'week04|1|cas_normal': ['num', 0.1, 'c26f04be80ff2670'],
    'week04|1|cas_rain': ['num', 0.1, '68e482a122b3c0c2'],
    'week04|1|reg_diff': ['num', 0.1, 'c25310d9a7f1f160'],
    'week04|1|reg_normal': ['num', 0.1, '3a2ce7a92f671ed1'],
    'week04|1|reg_rain': ['num', 0.1, 'ca5eeb3f65aae2f5'],
    'week04|2|cas_beta': ['num', 0.1, '09e9ca9e9ab07860'],
    'week04|2|reg_beta': ['num', 0.1, 'cdd6c575f9508284'],
    'week04|3|cas_high': ['num', 0.1, 'd6407ff4e5b07e76'],
    'week04|3|cas_low': ['num', 0.1, '1751c9943b6c108d'],
    'week04|3|reg_high': ['num', 0.1, '4f32326a85385ece'],
    'week04|3|reg_low': ['num', 0.1, '4faacf6e7828c4e2'],
    'week04|4|cas_adj_ratio': ['num', 0.001, 'ed362efef2ac508a'],
    'week04|4|cas_naive_ratio': ['num', 0.001, '6256be72652eff42'],
    'week04|4|reg_adj_ratio': ['num', 0.001, '2ecc4d7d2150dc15'],
    'week04|4|reg_naive_ratio': ['num', 0.001, 'ff11b84848a9fb53'],
    'week05|1|cost_all_val': ['int', None, 'f5b1aa205db1bae0'],
    'week05|1|cost_half_val': ['int', None, '1685004e1a036a6f'],
    'week05|1|cost_none_val': ['int', None, '15ace6debc27b2db'],
    'week05|1|fn_half': ['int', None, '7a3dab83caaafc4e'],
    'week05|1|fp_half': ['int', None, '5a7aec0901bb3ada'],
    'week05|2|chosen_costs#0': ['num', 1.0, '0a880c18b5ef8894'],
    'week05|2|chosen_costs#1': ['num', 1.0, '6924a38437d64048'],
    'week05|2|chosen_costs#2': ['num', 1.0, '88bbed5153f48155'],
    'week05|2|chosen_costs#3': ['num', 1.0, '2b786d8722e26a25'],
    'week05|2|chosen_thresholds#0': ['num', 1e-06, 'd3c1b5123e1eaa31'],
    'week05|2|chosen_thresholds#1': ['num', 1e-06, 'e5435506c1e37269'],
    'week05|2|chosen_thresholds#2': ['num', 1e-06, '2a6dfe234c8a76af'],
    'week05|2|chosen_thresholds#3': ['num', 1e-06, 'da0c3739e5dcdf74'],
    'week05|2|selected': ['rows', None, '26716c59075ccf4a'],
    'week05|3|evaluated': ['rows', None, '3177e51b72cb8c49'],
    'week05|3|test_costs#0': ['num', 1.0, '20ad2c17d387eba0'],
    'week05|3|test_costs#1': ['num', 1.0, '8d734f63f5f2f737'],
    'week05|3|test_costs#2': ['num', 1.0, 'd5541e1d046a80f2'],
    'week05|3|test_costs#3': ['num', 1.0, '17f8f175cf63ea79'],
    'week05|3|test_fn#0': ['num', 1.0, '38ea9bedd7744f4b'],
    'week05|3|test_fn#1': ['num', 1.0, 'daf78ce09fd175d2'],
    'week05|3|test_fn#2': ['num', 1.0, '8b4688ff144fe9d9'],
    'week05|3|test_fn#3': ['num', 1.0, 'a06413295e30abe2'],
    'week05|3|test_fp#0': ['num', 1.0, '59133eeb53809f8d'],
    'week05|3|test_fp#1': ['num', 1.0, '9a4c93657badb21f'],
    'week05|3|test_fp#2': ['num', 1.0, '6bc696c7e0c6d51b'],
    'week05|3|test_fp#3': ['num', 1.0, 'c5620698027070f1'],
    'week05|4|cost_ratios#0': ['num', 0.001, 'aafefd309719bd6f'],
    'week05|4|cost_ratios#1': ['num', 0.001, '5183d13a90095055'],
    'week05|4|cost_ratios#2': ['num', 0.001, 'f1519d489796dde6'],
    'week05|4|cost_ratios#3': ['num', 0.001, '99e27eb04d343434'],
    'week05|4|half_test_costs#0': ['num', 1.0, 'e06787fe70177e5c'],
    'week05|4|half_test_costs#1': ['num', 1.0, '1a71daef42c31366'],
    'week05|4|half_test_costs#2': ['num', 1.0, 'e2adc98f164e26cd'],
    'week05|4|half_test_costs#3': ['num', 1.0, '05b316f52eace5e3'],
    'week06|1|confirmed_rate#multiple': ['num', 0.001, 'f90739e98d0f7fa9'],
    'week06|1|confirmed_rate#single': ['num', 0.001, 'a901a45f415772c2'],
    'week06|1|group_counts#multiple': ['int', None, '916dc7d3050c23cc'],
    'week06|1|group_counts#single': ['int', None, 'e7a6f1b3da973bdb'],
    'week06|2|loop_cases#multiple': ['int', None, 'efe1d7bcd9428612'],
    'week06|2|loop_cases#single': ['int', None, '60a99938635d9c17'],
    'week06|2|loop_counts#multiple': ['int', None, 'afde1964a1ade711'],
    'week06|2|loop_counts#single': ['int', None, '730d598aeb855d55'],
    'week06|2|loops_per_case#multiple': ['num', 0.001, '0fa891a6fe20f056'],
    'week06|2|loops_per_case#single': ['num', 0.001, 'be2463784dbe992d'],
    'week06|3|phase_counts#multiple': ['int', None, '5fe0aa3e2e373b41'],
    'week06|3|phase_counts#single': ['int', None, 'f17201c7e08d90ae'],
    'week06|3|phase_mean#multiple': ['num', 0.01, '3061ee5398174cd0'],
    'week06|3|phase_mean#single': ['num', 0.01, '9d7567c8e7c4099d'],
    'week06|3|phase_median#multiple': ['num', 0.01, '5f400a9d055ffc3d'],
    'week06|3|phase_median#single': ['num', 0.01, '46a4da410a100cc0'],
    'week06|4|accepted_counts#multiple': ['int', None, 'eb922790f8c53803'],
    'week06|4|accepted_counts#single': ['int', None, '4670cf749eeb279d'],
    'week06|4|incomplete_rate#multiple': ['num', 0.001, 'a98f8ec6b4b982e2'],
    'week06|4|incomplete_rate#single': ['num', 0.001, '733623badb94183a'],
    'week06|4|mean_days#multiple': ['num', 0.01, 'ba7eef6c8cb72198'],
    'week06|4|mean_days#single': ['num', 0.01, '20c5473b217b32cc'],
    'week06|4|mean_rounds#multiple': ['num', 0.01, 'fed8e7fdd6cfbaa0'],
    'week06|4|mean_rounds#single': ['num', 0.01, '23f7c30403e47ab2'],
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

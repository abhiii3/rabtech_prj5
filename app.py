"""Executive KPI Dashboard - run with: streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Executive KPI Dashboard", layout="wide")

# ---------- Data (synthetic; swap in your own orders/customers tables) ----------
STATES = ["CA", "TX", "NY", "FL", "IL", "WA", "OH", "GA", "CO", "AZ", "MA", "NC"]
CATS = ["Electronics", "Apparel", "Home", "Beauty", "Sports"]
CHANNELS = {"Paid Search": 62, "Social": 48, "Email": 15, "Referral": 22, "Organic": 8}


@st.cache_data
def load_data():
    rng = np.random.default_rng(42)
    n = 4000
    start = pd.Timestamp("2024-01-01")
    cust = pd.DataFrame({
        "customer_id": range(n),
        "signup": start + pd.to_timedelta(rng.integers(0, 700, n), unit="D"),
        "state": rng.choice(STATES, n, p=np.array([14, 11, 11, 9, 8, 7, 6, 6, 6, 6, 8, 8]) / 100),
        "channel": rng.choice(list(CHANNELS), n, p=[.3, .25, .15, .15, .15]),
    })
    cust["cac"] = cust.channel.map(CHANNELS) * rng.uniform(0.8, 1.2, n)
    life = rng.geometric(0.04, n)  # monthly churn hazard ~4%
    cust["churn_date"] = cust.signup + pd.to_timedelta(life * 30, unit="D")
    cust.loc[cust.churn_date > pd.Timestamp("2025-12-31"), "churn_date"] = pd.NaT

    rows = []
    end = pd.Timestamp("2025-12-31")
    for r in cust.itertuples():
        stop = r.churn_date if pd.notna(r.churn_date) else end
        for m in pd.date_range(r.signup, stop, freq="30D"):
            for _ in range(rng.poisson(0.9)):
                rows.append((r.customer_id, m + pd.Timedelta(days=int(rng.integers(0, 30))),
                             rng.choice(CATS), rng.gamma(4, 18) + 10))
    orders = pd.DataFrame(rows, columns=["customer_id", "date", "category", "revenue"])
    orders = orders[orders.date <= end].merge(cust[["customer_id", "state", "channel"]], on="customer_id")
    return orders, cust


orders, cust = load_data()

# ---------- Slicers ----------
st.sidebar.header("Filters")
dmin, dmax = orders.date.min().date(), orders.date.max().date()
d_from, d_to = st.sidebar.date_input("Date range", (dmin, dmax), dmin, dmax)
states = st.sidebar.multiselect("State", STATES, STATES)
cats = st.sidebar.multiselect("Category", CATS, CATS)
chans = st.sidebar.multiselect("Channel", list(CHANNELS), list(CHANNELS))
gran = st.sidebar.radio("Time granularity", ["Month", "Quarter", "Year"], index=0, horizontal=True)
years = sorted(orders.date.dt.year.unique())
drill_year = st.sidebar.selectbox("Drill into year", ["All"] + years)

s, e = pd.Timestamp(d_from), pd.Timestamp(d_to)
if drill_year != "All":
    s, e, gran = pd.Timestamp(f"{drill_year}-01-01"), pd.Timestamp(f"{drill_year}-12-31"), "Month"

C = cust[cust.state.isin(states) & cust.channel.isin(chans)]
O = orders[orders.state.isin(states) & orders.channel.isin(chans) & orders.category.isin(cats)]


def kpis(a, b):
    o = O[(O.date >= a) & (O.date <= b)]
    rev = o.revenue.sum()
    new = C[(C.signup >= a) & (C.signup <= b)]
    base = C[(C.signup < a) & (C.churn_date.isna() | (C.churn_date >= a))]
    churn = base.churn_date.between(a, b).sum() / max(len(base), 1)
    return {"rev": rev, "aov": rev / max(len(o), 1), "cac": new.cac.mean() if len(new) else 0, "churn": churn * 100}


cur = kpis(s, e)
prev = kpis(s - (e - s) - pd.Timedelta(days=1), s - pd.Timedelta(days=1))


def delta(k):
    return f"{(cur[k] - prev[k]) / prev[k] * 100:+.1f}% vs prior period" if prev[k] else None


# ---------- Layout: headline KPIs -> trend -> geo + breakdown -> drill-down ----------
st.title("Executive KPI Dashboard")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Revenue", f"${cur['rev']:,.0f}", delta("rev"))
k2.metric("Avg Order Value", f"${cur['aov']:,.2f}", delta("aov"))
k3.metric("Customer Acquisition Cost", f"${cur['cac']:,.2f}", delta("cac"), delta_color="inverse")
k4.metric("Churn Rate", f"{cur['churn']:.1f}%", delta("churn"), delta_color="inverse")

Ow = O[(O.date >= s) & (O.date <= e)].copy()
freq = {"Month": "M", "Quarter": "Q", "Year": "Y"}[gran]
Ow["period"] = Ow.date.dt.to_period(freq).dt.to_timestamp()
trend = Ow.groupby(["period", "category"], as_index=False).revenue.sum()
st.subheader(f"Revenue trend by {gran.lower()}")
st.plotly_chart(px.area(trend, x="period", y="revenue", color="category",
                        labels={"period": "", "revenue": "Revenue ($)"}), use_container_width=True)

g1, g2 = st.columns([3, 2])
geo = Ow.groupby("state", as_index=False).revenue.sum()
g1.subheader("Revenue heatmap by state")
g1.plotly_chart(px.choropleth(geo, locations="state", locationmode="USA-states", scope="usa",
                              color="revenue", color_continuous_scale="Blues"), use_container_width=True)

dim = g2.selectbox("Break down by", ["category", "channel", "state"])
g2.subheader(f"Revenue by {dim}")
bd = Ow.groupby(dim, as_index=False).revenue.sum().sort_values("revenue")
g2.plotly_chart(px.bar(bd, x="revenue", y=dim, orientation="h"), use_container_width=True)

st.subheader("Categorical drill-down")
d1, d2 = st.columns(2)
member = d1.selectbox(f"Select a {dim}", sorted(Ow[dim].unique()))
sub = d2.selectbox("Then split by", [d for d in ["category", "channel", "state"] if d != dim])
det = Ow[Ow[dim] == member].groupby(sub, as_index=False).agg(revenue=("revenue", "sum"), orders=("revenue", "size"))
det["aov"] = det.revenue / det.orders
st.dataframe(det.sort_values("revenue", ascending=False).style.format(
    {"revenue": "${:,.0f}", "aov": "${:,.2f}"}), use_container_width=True)
st.download_button("Download filtered orders (CSV)", Ow.to_csv(index=False), "filtered_orders.csv")

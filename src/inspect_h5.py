"""Probe a LargeST yearly HDF5 file's internal layout so we can build a
memory-bounded loader. Prints pandas key, format, index span, columns."""
import sys
import h5py
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "data/ca/ca_his_raw_2017.h5"

print(f"== h5py tree: {path} ==")
def show(name, obj):
    kind = "grp" if isinstance(obj, h5py.Group) else f"dset shape={obj.shape} dtype={obj.dtype}"
    print(f"  {name}: {kind}")
with h5py.File(path, "r") as f:
    f.visititems(show)
    print("  root attrs:", dict(f.attrs))

print("\n== pandas HDFStore ==")
with pd.HDFStore(path, "r") as st:
    print("  keys:", st.keys())
    for k in st.keys():
        storer = st.get_storer(k)
        print(f"  {k}: format={storer.format_type} nrows={getattr(storer,'nrows',None)}")

# Peek a tiny slice without loading everything (fixed format: start/stop rows)
print("\n== head peek ==")
try:
    df = pd.read_hdf(path, start=0, stop=3)
    print("  shape(head):", df.shape)
    print("  index[:3]:", list(df.index[:3]))
    print("  index dtype:", df.index.dtype)
    print("  first 5 cols:", list(df.columns[:5]), "... dtype:", df.columns.dtype)
    print("  values dtype:", df.values.dtype)
except Exception as e:
    print("  head peek failed:", e)

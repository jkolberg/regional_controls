from util import Util

counties = util.settings['counties']
counties = [int('53'+c) for c in counties]

def filter_lodes(df):
    geocode_col = [c for c in df.columns if c.startswith(('w_', 'h_')) and 'geocode' in c][0]
    df = df.rename(columns={geocode_col: 'geocode'})
    df['county_id'] = df['geocode'].astype(str).str[:5].astype(int)
    df = df.query('county_id.isin(@counties)')
    df = df.groupby('county_id').sum()
    df = df[[c for c in df.columns if c.startswith('CNS')]]
    return df

all_rac = filter_lodes(pd.read_csv('https://lehd.ces.census.gov/data/lodes/LODES8/wa/rac/wa_rac_S000_JT00_2023.csv.gz', compression='gzip'))
all_wac = filter_lodes(pd.read_csv('https://lehd.ces.census.gov/data/lodes/LODES8/wa/wac/wa_wac_S000_JT00_2023.csv.gz', compression='gzip'))

prim_rac = filter_lodes(pd.read_csv('https://lehd.ces.census.gov/data/lodes/LODES8/wa/rac/wa_rac_S000_JT01_2023.csv.gz', compression='gzip'))
prim_wac = filter_lodes(pd.read_csv('https://lehd.ces.census.gov/data/lodes/LODES8/wa/wac/wa_wac_S000_JT01_2023.csv.gz', compression='gzip'))

priv_rac = filter_lodes(pd.read_csv('https://lehd.ces.census.gov/data/lodes/LODES8/wa/rac/wa_rac_S000_JT02_2023.csv.gz', compression='gzip'))
priv_wac = filter_lodes(pd.read_csv('https://lehd.ces.census.gov/data/lodes/LODES8/wa/wac/wa_wac_S000_JT02_2023.csv.gz', compression='gzip'))

priv_primary_rac = filter_lodes(pd.read_csv('https://lehd.ces.census.gov/data/lodes/LODES8/wa/rac/wa_rac_S000_JT03_2023.csv.gz', compression='gzip'))
priv_primary_wac = filter_lodes(pd.read_csv('https://lehd.ces.census.gov/data/lodes/LODES8/wa/wac/wa_wac_S000_JT03_2023.csv.gz', compression='gzip'))

def calcualate_gov_primary(all,prim,priv,priv_primary):
    # government
    gov = all - priv
    # government primary
    gov_primary = prim - priv_primary

    return gov.sum(axis=1), gov_primary.sum(axis=1)

gov_rac, gov_primary_rac = calcualate_gov_primary(all_rac, prim_rac, priv_rac, priv_primary_rac)
gov_wac, gov_primary_wac = calcualate_gov_primary(all_wac, prim_wac, priv_wac, priv_primary_wac)

primary_gov_workers_rate = gov_primary_rac / gov_rac
primary_private_workers_rate = (priv_primary_rac / priv_rac).unstack()

industry_xwalk = util.get_table('industry_crosswalk').dropna(subset=['cns']).drop_duplicates(subset=['cns'])
cns_to_industry = industry_xwalk.set_index('cns')['industry'].astype(str).to_dict()
primary_private_workers_rate = primary_private_workers_rate.rename(cns_to_industry, level=0)
primary_private_workers_rate.index.names = [ 'industry','county_id']
primary_private_workers_rate.loc['98'] = primary_gov_workers_rate
primary_workers_rates = primary_private_workers_rate
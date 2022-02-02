# Import packages that are not part of base Python
import os, pathlib, google, numpy as np, pandas as pd
from google.cloud import bigquery

# bigquery_storage is not built into Vertex AI, so we install if necessary.
# try-except an extrememly useful python feature.  Here, we first try to import from bigquery_storage
# If that works, program moves on.  If that fails because it's not installed, we pip install it from https://pypi.org/.
try:
    from google.cloud.bigquery_storage import BigQueryReadClient
except:
    os.system('pip install --upgrade google-cloud-bigquery-storage')
    from google.cloud.bigquery_storage import BigQueryReadClient
    
# Create connection to BigQuery
cred, proj = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
bqclient   = bigquery.Client(credentials=cred, project=proj)
# change this to your BigQuery proj_id
proj_id = 'sublime-bongo-332800'
# change this to your dataset name
dataset = f'{proj_id}.COVID_Project'

# Define useful functions to interact with BigQuery
def get_cols(tbl):
    """Get list of columns on tbl"""
    t = bqclient.get_table(tbl)
    return [s.name for s in t.schema]

def run_query(query):
    """Run sql query and return pandas dataframe of results, if any"""
    res = bqclient.query(query).result()
    try:
        return res.to_dataframe()
    except:
        return True
    
def head(tbl, rows=10):
    """Display the top rows of tbl"""
    query = f'select * from {tbl} limit {rows}'
    df = run_query(query)
    print(df)
    return df
    
def delete_table(tbl):
    """Delete tbl if it exists"""
    query = f'drop table {tbl}'
    try:
        run_query(query)
    except google.api_core.exceptions.NotFound:
        pass

def load_table(tbl, df=None, query=None, file=None, overwrite=True, preview_rows=0):
    """Load data into tbl either from a pandas dataframe, sql query, or local csv file"""
    
    if overwrite:
        delete_table(tbl)
    
    if df is not None:
        job = bqclient.load_table_from_dataframe(df, tbl).result()
    elif query is not None:
        job = bqclient.query(query, job_config=bigquery.QueryJobConfig(destination=tbl)).result()
    elif file is not None:
        with open(file, mode='rb') as f:
            job = bqclient.load_table_from_file(f, tbl, job_config=bigquery.LoadJobConfig(autodetect=True)).result()
    else:
        raise Exception('at least one of df, query, or file must be specified')
    
    if preview_rows > 0:
        head(tbl, preview_rows)
    return tbl

def subquery(query, indents=1):
    s = '\n' + indents * '    '
    return query.strip().replace('\n', s)

def print_query(query, final_only=False):
    if final_only:
        print(query[-1])
    else:
        for k, q in enumerate(query):
            print(f'stage {k}')
            print(q)
            print('===============================================================================')
            
#######################################################################################################            
# Additional packages and functions to be imported
#######################################################################################################

# Function to analyze multiple output objects at once
from IPython.display import display

# Consistent joining of shape files with project dataset is timely and often causes program to crash.
# The following provides the same shape information (for use in plotly visuals) in a much more efficient manner.
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})

# Plotly. express package and functions to create visualizations
# If reader does not have plotly installed, uncomment and run line below.
#!pip install plotly
import plotly.express as px
from plotly.subplots import make_subplots

#######################################################################################################
# Create function to construct project datasets
#######################################################################################################

def create_datasets():
    # Subquery used to create the primary project dataset
    primary = f'''
    `bigquery-public-data.covid19_nyt.mask_use_by_county`
    '''

    # Subquery used to create the first alternative project dataset (Alt1)
    alt1 = f'''
    (
        SELECT
            county_fips_code,
            no_mask,
            sometimes_mask,
            always_mask 
        FROM( 
            SELECT
                *,
                never AS no_mask,
                rarely+sometimes+frequently AS sometimes_mask,
                always AS always_mask
            FROM
                `bigquery-public-data.covid19_nyt.mask_use_by_county` 
        )
    )
    '''

    # Subquery used to create the second alternative project dataset (Alt2)
    alt2 = f'''
    (
        SELECT
            county_fips_code,
            no_mask,
            mask 
        FROM(
            SELECT
                *,
                never+rarely AS no_mask,
                sometimes+frequently+always AS mask
            FROM
                `bigquery-public-data.covid19_nyt.mask_use_by_county` 
        )
    )
    '''

    # Subquery used to create the third alternative project dataset (Alt3)
    alt3 = f'''
    (
        SELECT
            county_fips_code,
            no_mask,
            mask 
        FROM( 
            SELECT
                *,
                never+rarely+sometimes AS no_mask,
                frequently+always AS mask 
            FROM
                `bigquery-public-data.covid19_nyt.mask_use_by_county` 
        )
    )
    '''

    # List of subqueries (above) used to create each project dataset
    datasets = (primary,alt1,alt2,alt3)

    # List to store each dataframe output generated by for loop (below)
    df_list = list()

    for data in datasets:
        # Query that constructs project dataset
        query = f'''
            SELECT
                * EXCEPT( 
                    total_pop,
                    male_pop,
                    male_pop_50_and_over,
                    female_pop,
                    female_pop_50_and_over,
                    amerindian_pop,
                    white_pop,
                    black_pop,
                    asian_pop,
                    hispanic_pop,
                    other_not_disclosed,
                    pop_determined_poverty_status 
                ),
                --Express demographic data as population proportions,rather than raw numbers
                male_pop/total_pop AS male_pop,
                female_pop/total_pop AS female_pop,
                (male_pop_50_and_over + female_pop_50_and_over)/total_pop AS pop_50_and_over,
                amerindian_pop/total_pop AS amerindian_pop,
                white_pop/total_pop AS white_pop,
                black_pop/total_pop AS black_pop,
                asian_pop/total_pop AS asian_pop,
                hispanic_pop/total_pop AS hispanic_pop,
                other_not_disclosed/total_pop AS other_not_disclosed,
                pop_determined_poverty_status/total_pop AS pop_determined_poverty_status
            FROM(
                SELECT
                    * EXCEPT(
                        male_50_to_54,
                        male_55_to_59,
                        male_60_to_61,
                        male_62_to_64,
                        male_65_to_66,
                        male_67_to_69,
                        male_70_to_74,
                        male_75_to_79,
                        male_80_to_84,
                        male_85_and_over,
                        female_50_to_54,
                        female_55_to_59,
                        female_60_to_61,
                        female_62_to_64,
                        female_65_to_66,
                        female_67_to_69,
                        female_70_to_74,
                        female_75_to_79,
                        female_80_to_84,
                        female_85_and_over
                    ),
                    (new_cases_mu - new_cases_per_10k_difference)/new_cases_sigma as new_cases_z,
                    (male_50_to_54+male_55_to_59+
                        male_60_to_61+male_62_to_64+male_65_to_66+male_67_to_69+
                        male_70_to_74+male_75_to_79+
                        male_80_to_84+male_85_and_over) AS male_pop_50_and_over,
                    (female_50_to_54+female_55_to_59+
                        female_60_to_61+female_62_to_64+female_65_to_66+female_67_to_69+
                        female_70_to_74+female_75_to_79+
                        female_80_to_84+female_85_and_over) AS female_pop_50_and_over,
                FROM(
                    SELECT
                        c.county_fips_code,
                        c.county_name,
                        c.new_cases_per_10k_difference,
                        m.* EXCEPT(county_fips_code),
                        p.school_closing,
                        p.workplace_closing,
                        p.cancel_public_events,
                        p.restrictions_on_gatherings,
                        p.close_public_transit,
                        p.stay_at_home_requirements,
                        p.restrictions_on_internal_movement,
                        p.international_travel_controls,
                        p.stringency_index,
                        c.state,
                        d.region,
                        d.Northeast,
                        d.Midwest,
                        d.South,
                        d.West,
                        d.pop_density,
                        r.metropolitan,
                        avg(new_cases_per_10k_difference) over (partition by null) as new_cases_mu,
                        stddev_pop(new_cases_per_10k_difference) over (partition by null) as new_cases_sigma,
                        a.total_pop,
                        a.male_pop,
                        a.male_50_to_54,
                        a.male_55_to_59,
                        a.male_60_to_61,
                        a.male_62_to_64,
                        a.male_65_to_66,
                        a.male_67_to_69,
                        a.male_70_to_74,
                        a.male_75_to_79,
                        a.male_80_to_84,
                        a.male_85_and_over,
                        a.female_pop,
                        a.female_50_to_54,
                        a.female_55_to_59,
                        a.female_60_to_61,
                        a.female_62_to_64,
                        a.female_65_to_66,
                        a.female_67_to_69,
                        a.female_70_to_74,
                        a.female_75_to_79,
                        a.female_80_to_84,
                        a.female_85_and_over,
                        a.amerindian_pop,
                        a.white_pop,
                        a.black_pop,
                        a.asian_pop,
                        a.hispanic_pop,
                        (total_pop-(white_pop+black_pop+asian_pop+hispanic_pop+amerindian_pop)) AS other_not_disclosed,
                        a.median_income,
                        a.pop_determined_poverty_status
                    FROM {data} AS m
                    JOIN(
                        select
                            county_fips_code,
                            county_name,
                            state,
                            new_cases_per_10k_difference
                        from(
                            select
                                *,
                                avg_new_cases_per_10k - avg_new_cases_per_10k_prev as new_cases_per_10k_difference
                            from(
                                select
                                    * except(new_cases_per_10k),
                                    lag(avg_new_cases_per_10k, 1) OVER (partition by county_fips_code ORDER BY date) as avg_new_cases_per_10k_prev
                                from(
                                    Select 
                                        *,
                                        avg(new_cases_per_10k) OVER (partition by county_fips_code, week ORDER BY date) as avg_new_cases_per_10k
                                    from( 
                                        select 
                                            * except(new_cases, total_pop),
                                            (new_cases*10000)/total_pop as new_cases_per_10k
                                        from(
                                            select
                                                county_fips_code,
                                                county_name,
                                                state,
                                                date,
                                                week,
                                                new_cases,
                                                c.total_pop
                                            from(
                                                select
                                                    *
                                                from(
                                                    select
                                                        *,
                                                        extract(week(wednesday) from date) as week,
                                                        confirmed_cases - cases_previous_day as new_cases
                                                    from(
                                                        select
                                                            * EXCEPT(deaths),
                                                            lag(confirmed_cases, 1) OVER (partition by county_fips_code ORDER BY date) as cases_previous_day
                                                        from 
                                                            `bigquery-public-data.covid19_usafacts.summary`
                                                        where
                                                            date between '2020-06-09' and '2020-06-16' or
                                                            date between '2020-07-14' and '2020-07-21'
                                                    )
                                                )
                                                where 
                                                    week in (24,29)
                                            )
                                            LEFT JOIN 
                                                `bigquery-public-data.census_bureau_acs.county_2018_5yr` as c
                                            ON
                                                county_fips_code = c.geo_id
                                            order by
                                                county_name
                                        )
                                        order by county_name, week
                                    )
                                )
                                where 
                                    date = '2020-06-16' or
                                    date = '2020-07-21'
                            )
                        )
                        where new_cases_per_10k_difference is not null
                        order by county_name, state, week, date
                    ) as c
                    USING
                        (county_fips_code)
                    JOIN(
                        select
                            region_code,
                            max(school_closing) as school_closing,
                            max(workplace_closing) as workplace_closing,
                            max(cancel_public_events) as cancel_public_events,
                            max(restrictions_on_gatherings) as restrictions_on_gatherings,
                            max(close_public_transit) as close_public_transit,
                            max(stay_at_home_requirements) as stay_at_home_requirements,
                            max(restrictions_on_internal_movement) as restrictions_on_internal_movement,
                            max(international_travel_controls) as international_travel_controls,
                            max(stringency_index) as stringency_index,
                            max(testing_policy) as testing_policy
                        from(
                            select
                                *,
                                lag(stringency_index) over (partition by region_code, month order by date) as prev_stringency
                            from(
                                select
                                    *,
                                    extract(month from date) as month
                                from(
                                    SELECT
                                        region_code,
                                        date,
                                        school_closing,
                                        workplace_closing,
                                        cancel_public_events,
                                        restrictions_on_gatherings,
                                        close_public_transit,
                                        stay_at_home_requirements,
                                        restrictions_on_internal_movement,
                                        international_travel_controls,
                                        stringency_index,
                                        testing_policy
                                    FROM 
                                        `bigquery-public-data.covid19_govt_response.oxford_policy_tracker` 
                                    where
                                        SUBSTRING(region_code, 1, 2) = 'US' and 
                                        date in ('2020-06-09' , '2020-06-16' , '2020-07-14' , '2020-07-21')
                                    order by region_code, date asc
                                )
                            )
                        )
                        group by    
                            region_code
                        order by
                            region_code
                    ) as p
                    ON
                        SUBSTRING(p.region_code, 4, 2) = c.state
                    JOIN(
                        SELECT
                            *
                        FROM
                            `bigquery-public-data.census_bureau_acs.county_2018_5yr`
                        WHERE
                            median_income is not null
                    ) AS a
                    ON
                        a.geo_id = c.county_fips_code
                    JOIN( 
                        --For full documentation on this subquery,see "population_density Fixer"
                        SELECT
                            *,
                            CASE
                                WHEN state IN (09, 23, 25, 33, 44, 50, 34, 36, 42) THEN 1
                                ELSE
                            0
                            END AS Northeast,
                            CASE
                                WHEN state IN (18, 17, 26, 39, 55, 19, 20, 27, 29, 31, 38, 46) THEN 1
                                ELSE 0
                            END AS Midwest,
                            CASE
                                WHEN state IN (10, 11, 12, 13, 24, 37, 45, 51, 54, 01, 21, 28, 47, 05, 22, 40, 48) THEN 1
                                ELSE 0
                            END AS South,
                            CASE
                                WHEN state IN (04, 08, 16, 35, 30, 49, 32, 56, 02, 06, 15, 41, 53) THEN 1
                                ELSE 0
                            END AS West,
                            CASE
                                WHEN state IN (04, 08, 16, 35, 30, 49, 32, 56, 02, 06, 15, 41, 53) THEN 'West'
                                WHEN state IN (10, 11, 12, 13, 24, 37, 45, 51, 54, 01, 21, 28, 47, 05, 22, 40, 48) THEN 'South'
                                WHEN state IN (18, 17, 26, 39, 55, 19, 20, 27, 29, 31, 38, 46) THEN 'Midwest'
                                WHEN state IN (09, 23, 25, 33, 44, 50, 34, 36, 42) THEN 'Northeast'
                                ELSE NULL
                            END AS Region
                        FROM (
                            SELECT
                                *,
                                CAST(SUBSTRING(geoid,1,2) AS int) AS state
                            FROM (
                                SELECT
                                    CASE
                                        WHEN LENGTH(CAST(geoid AS string)) = 4 THEN CONCAT(0,GEOID)
                                        ELSE CAST(GEOID AS string)
                                    END AS geoid,
                                    B01001_calc_PopDensity AS pop_density
                                FROM
                                    `sublime-bongo-332800.COVID_Project.population_densities` 
                            )
                        )
                    ) AS d
                    ON
                        d.geoid = c.county_fips_code
                    JOIN(
                        --For full documentation on this table,see "metropolitan_counties Fixer" 
                        SELECT
                            CASE
                                WHEN LENGTH(CAST(fips_code AS string)) = 4 THEN CONCAT(0,fips_code)
                                ELSE CAST(fips_code AS string)
                            END AS fips_code,
                            CASE
                                WHEN _2013_code in (1,2,3,4) THEN 1
                                ELSE 0
                            END AS metropolitan
                        FROM
                            `sublime-bongo-332800.COVID_Project.metropolitan_counties` 
                    ) AS r
                    ON
                        r.fips_code = c.county_fips_code
                )
            )
            '''

        # Run query and save as dataframe
        df = pd.DataFrame(run_query(query))

        # Save dataframe to df_list created outside of loop
        df_list.append(df)

    return(df_list)


#######################################################################################################
# Save datasets
#######################################################################################################

# The query below constructs the dependent variable used in the regression analysis.
# As such, it is a vital query and will be made available to the reader via the function below
def show_new_cases():
    query = []
    
    query.append(f'''
    select
        * EXCEPT(deaths),
        lag(confirmed_cases, 1) OVER (partition by county_fips_code ORDER BY date) as cases_previous_day
    from 
        `bigquery-public-data.covid19_usafacts.summary`
    where
        date between '2020-06-09' and '2020-06-16' or
        date between '2020-07-14' and '2020-07-21'
    ''')
    
    query.append(f'''
    select
        *,
        extract(week(wednesday) from date) as week,
        confirmed_cases - cases_previous_day as new_cases
    from(
        {subquery(query[-1])}
    )
    ''')
    
    query.append(f'''
    select
        *
    from(
        {subquery(query[-1])}
    )

    ''')
    
    query.append(f'''
    select
        *
    from(
        {subquery(query[-1])}
    )
    where 
        week in (24,29)
    ''')
    
    query.append(f'''
    select
        county_fips_code,
        county_name,
        state,
        date,
        week,
        new_cases,
        c.total_pop
    from(
        {subquery(query[-1])}
    )
    LEFT JOIN 
        `bigquery-public-data.census_bureau_acs.county_2018_5yr` as c
    ON
        county_fips_code = c.geo_id
    order by
        county_name
    ''')
                 
    query.append(f'''
    select 
        * except(new_cases, total_pop),
        (new_cases*10000)/total_pop as new_cases_per_10k
    from(
        {subquery(query[-1])}
    )
    order by county_name, week
    ''')
    
    query.append(f'''
    Select 
        *,
        avg(new_cases_per_10k) OVER (partition by county_fips_code, week ORDER BY date) as avg_new_cases_per_10k
    from( 
        {subquery(query[-1])}
    )
    ''')
    
    query.append(f'''
    select
        * except(new_cases_per_10k),
        lag(avg_new_cases_per_10k, 1) OVER (partition by county_fips_code ORDER BY date) as avg_new_cases_per_10k_prev
    from(
        {subquery(query[-1])}
    )
    where 
        date = '2020-06-16' or
        date = '2020-07-21'
    ''')
    
    query.append(f'''
    select
        *,
        avg_new_cases_per_10k - avg_new_cases_per_10k_prev as new_cases_per_10k_difference
    from(
        {subquery(query[-1])}
    )
    ''')
    
    query.append(f'''
    select
        county_fips_code,
        county_name,
        state,
        new_cases_per_10k_difference
    from(
        {subquery(query[-1])}
    )
    where new_cases_per_10k_difference is not null
    order by county_name, state, week, date
    ''')
    
    print_query(query)

    
#######################################################################################################
# Save datasets
#######################################################################################################

# create_datasets() fuctions returns a list of 4 datasets
df_list = create_datasets()

# save each list item to correct project dataset variable
primary_df = df_list[0]
alt1_df = df_list[1]
alt2_df = df_list[2]
alt3_df = df_list[3]

#######################################################################################################
# Generate and display summary statistics for each dataset
#######################################################################################################

# Since datasets are identical, except for mask use variables, only summary statistics for mask usage variables
# will be shown for Alternative Models 1-3

# Since the categories shown represent population proportions that adhere to various levels of mask usage,
# the mean of each represents mask use proportions at the national level

# Force pandas to display all columns in dataframe
pd.set_option('display.max_columns', None)

# Display descriptive statistics
# print('Descriptive Statistics - Primary Model')
# display(primary_df.describe())
# print('\n'+'====================================================================='+'\n')
# print('Descriptive Statistics - Alternative Model 1')
# display(alt1_df[['no_mask','sometimes_mask','always_mask','new_cases_z']].describe())
# print('\n'+'====================================================================='+'\n')
# print('Descriptive Statistics - Alternative Model 2')
# display(alt2_df[['no_mask','mask']].describe())
# print('\n'+'====================================================================='+'\n')
# print('Descriptive Statistics - Alternative Model 3')
# display(alt3_df[['no_mask','mask']].describe())

#######################################################################################################
# Investigate distribution of intended dependent variable (new_cases_per_10k_difference)
#######################################################################################################

cases_hist = px.histogram(primary_df, 
                   x= 'new_cases_per_10k_difference', 
                   range_x= (-5,5),
                   title="Histogram of Y",
                   labels={"new_cases_per_10k_difference":"Difference in Average New Cases"}
                  )
#cases_chlor.show()

#######################################################################################################
#Visualize depdendent variable (in essence, change in weekly new COVID cases during observation period)
#######################################################################################################

cases_chlor = px.choropleth(primary_df, geojson=counties, locations='county_fips_code', color='new_cases_per_10k_difference',
                           color_continuous_scale="Blues",
                           range_color=(-7, 7),
                           scope="usa",
                           hover_name='county_name',
                           hover_data={'new_cases_z':':.4f',
                                       'region':True,
                                       'state':True,
                                       'new_cases_per_10k_difference':':.4f'
                                      },
                           labels={'county_fips_code':'FIPS code ',
                                   'new_cases_per_10k_difference':'Difference ',
                                   'new_cases_z': 'Difference (z-score) ',
                                   'region':'Region ',
                                   'state':'State '
                                  },
                            title="Differences in Average New Cases During Observation Period"
                          )
cases_chlor.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#cases_chlor.show()

#######################################################################################################
# Explore relationships between dependent variable, mask usage variables, and population density
#######################################################################################################

# While aim of project is to determine whether mask usage significantly effects
# the spread of COVID-19, one hypothesis is that mask usage is more important/impactful
# in densely populated/metropoltian areas.
# Population density has a large range, making meaningful visualization tricky. The code below
# determines various percentiles for population density to aid in removal of outliers.
# (Recall that values for 'pop_density' do not vary across datasets)

# Initialize empty dictionary to store percentiles
density_percentiles = {}

for i in range(0,105,5):
    p = np.percentile(primary_df['pop_density'],i)
    density_percentiles[str(i)] = p

#density_percentiles output
#density_percentiles

# Since aim is to determine whether population density and mask effectiveness are related,
# we do not want to lose too many records with high population densities.
# Dropping the top 10% seems to produce decent results.

#Initialize empty list to store modified datasets (bottom 90th percentile of population densities)
modified_df_list = [] 

#Loop to drop records in top top 10th percentile of population densities for each dataset
for d in df_list:
    modified_df = d[d['pop_density']<=153]
    modified_df_list.append(modified_df)

# Generate scatter plots for each modified dataset
fig0 = px.scatter(modified_df_list[0], 
                 y=['never','rarely','sometimes','frequently','always'], 
                 x='new_cases_per_10k_difference',
                 range_x=(-4,7),
                 size= 'pop_density',
                 title= 'Primary Model'
                )
fig0.add_vline(x=0, line_width=2)

fig1 = px.scatter(modified_df_list[1], 
                 y=['no_mask','sometimes_mask','always_mask'], 
                 x='new_cases_per_10k_difference',
                 range_x=(-4,7),
                 size= 'pop_density',
                 title= 'Alternative Model 1'
                )
fig1.add_vline(x=0, line_width=2)

fig2 = px.scatter(modified_df_list[2], 
                 y=['no_mask','mask'], 
                 x='new_cases_per_10k_difference',
                 range_x=(-4,7),
                 size = 'pop_density',
                 title = 'Alternative Model 2'
                )
fig2.add_vline(x=0, line_width=2)

fig3 = px.scatter(modified_df_list[3], 
                 y=['no_mask','mask'], 
                 x='new_cases_per_10k_difference',
                 range_x=(-4,7),
                 size = 'pop_density',
                 title = 'Alternative Model 3'
                )
fig3.add_vline(x=0, line_width=2)

# Display scatterplots
# display(fig0)
# print('\n'+'====================================================================='+'\n')
# display(fig1)
# print('\n'+'====================================================================='+'\n')
# display(fig2)
# print('\n'+'====================================================================='+'\n')
# display(fig3)

# Use for loop to generate density heatmaps for each mask use variable in each dataset

# Primary dataset
primary_categories = ['never','rarely','sometimes','frequently','always']

primary_plots = list()

for cat in primary_categories:
    fig = px.density_heatmap(primary_df, 
                             x= cat, 
                             y="new_cases_per_10k_difference", 
                             range_y=(-3,3),
                             nbinsx=40,
                             nbinsy=400
                             )
    fig.update_layout(title = 'Primary Model')
    primary_plots.append(fig)
    
# Alt1 dataset
alt1_categories = ['no_mask','sometimes_mask','always_mask']

alt1_plots = list()

for cat in alt1_categories:
    fig = px.density_heatmap(alt1_df, 
                             x= cat, 
                             y="new_cases_per_10k_difference", 
                             range_y=(-3,3),
                             nbinsx=40,
                             nbinsy=400
                             )
    fig.update_layout(title = 'Alternative Model 1')
    alt1_plots.append(fig)
    
# Alt2 dataset
alt2_categories = ['no_mask','mask']

alt2_plots = list()

for cat in alt2_categories:
    fig = px.density_heatmap(alt2_df, 
                             x= cat, 
                             y="new_cases_per_10k_difference", 
                             range_y=(-3,3),
                             nbinsx=40,
                             nbinsy=400
                             )
    fig.update_layout(title = 'Alternative Model 2')
    alt2_plots.append(fig)

# Alt3 dataset
alt3_categories = ['no_mask','mask']

alt3_plots = list()


for cat in alt3_categories:
    fig = px.density_heatmap(alt3_df, 
                             x= cat, 
                             y="new_cases_per_10k_difference", 
                             range_y=(-3,3),
                             nbinsx=40,
                             nbinsy=400
                             )
    fig.update_layout(title = 'Alternative Model 3')
    alt3_plots.append(fig)    
    
#plot all heatmaps (if desired, comment out selected models)
# for i in range(0,len(primary_plots)):
#     display(primary_plots[i])
# for i in range(0,len(alt1_plots)):
#     display(alt1_plots[i])
# for i in range(0,len(alt2_plots)):
#     display(alt2_plots[i])    
# for i in range(0,len(alt3_plots)):
#     display(alt3_plots[i])


#######################################################################################################
# Explore results dataset
#######################################################################################################

def create_results_dataset():
    query = f'''
        select
            *
        from
            `sublime-bongo-332800.COVID_Project.regression_results`
        '''
    
    # Run query and save as dataframe
    df = pd.DataFrame(run_query(query))
    
    return(df)

regression_results = create_results_dataset()
#regression_results.head()

fig = px.choropleth(regression_results, geojson=counties, locations='fips_code', color='primary_rsq',
                           color_continuous_scale="blues",
                           scope="usa",
                           hover_name = 'state',
                           hover_data={'primary_rsq':':.4f',
                                       'primary_rsq':':.4f'
                                      },
                           labels={'fips_code':'FIPS code ',
                                   'primary_rsq':'R-squared '
                                  },
                           range_color=(-3,1)
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#fig.show()

# Explore percentiles of variable to determine optimal scaling on chloropleth map

rsq_percentiles={}

for i in range(0,105,5):
    p = np.percentile(regression_results['primary_step_rsq'],i)
    rsq_percentiles[str(i)] = p
    
rsq_percentiles

#Create chloropleth map of R-squared values for best model
best_model_chlor = px.choropleth(regression_results, geojson=counties, locations='fips_code', color='primary_step_rsq',
                           color_continuous_scale="Blues",
                           scope="usa",
                           hover_name = 'state',
                           hover_data={'primary_step_rsq':':.4f'
                                      },
                           labels={'fips_code':'FIPS code ',
                                   'primary_step_rsq':'R-squared '
                                  },
                           range_color=(-3,1)
                          )
best_model_chlor.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#fig.show()

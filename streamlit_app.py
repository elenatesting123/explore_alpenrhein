import streamlit as st

st.set_page_config(layout="wide")

import netCDF4
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
# from utils import update_data_obs_nonetcdf,fetch_pred
# from config_alpenrhein import measurements
import glob
import json
from datetime import timedelta,datetime,date
import numpy as np
import requests
from requests.auth import HTTPBasicAuth


def fetch_obs(location_id, product_id, datetime_start, datetime_end):
    print(f'Fetching obs between {datetime_start} and {datetime_end}')
    string_date = f"?lb={datetime_start.year}-{datetime_start.month:02d}-{datetime_start.day:02d}T{datetime_start.hour:02d}%3A{datetime_start.minute:02d}%3A{datetime_start.second:02d}.000Z&ub={datetime_end.year}-{datetime_end.month:02d}-{datetime_end.day:02d}T{datetime_end.hour:02d}%3A{datetime_end.minute:02d}%3A{datetime_end.second:02d}.000Z"

    url = "https://www.gin5.admin.ch/gin/machine-api/api/value/station/measurement/"+location_id+"/"+product_id+"/"+"result"+string_date
    response_obs = requests.get(url, auth=HTTPBasicAuth(st.session_state['username'],st.session_state['password']))
    try:
        df_obs = pd.DataFrame(response_obs.json())
        
        if len(df_obs)==1 and df_obs['noData']:
            print('no data for ', datetime_start, datetime_end)
            return None
        elif 'time' in df_obs.columns:
            df_obs['time'] = pd.to_datetime(df_obs['time'], utc=True)
            df_obs['time'] = df_obs['time'].dt.tz_convert(None)
            
            # Set 'time' as the index and keep only the 'value' column
            df_obs = df_obs.set_index("time")[["value"]]

            # Optional: sort by time just in case
            df_obs = df_obs.sort_index()
            print('after:', len(df_obs))
        else:
            print('no data for ', datetime_start, datetime_end)
            return None
    except ValueError:
        print('NO FILE FOUND')
        print(url)
        print(response_obs.text)
        return None

    return df_obs

def fetch_pred(location_id, product_id, datetime_start, datetime_end):
    string_date = f"?lb={datetime_start.year}-{datetime_start.month:02d}-{datetime_start.day:02d}T{datetime_start.hour:02d}%3A{datetime_start.minute:02d}%3A{datetime_start.second:02d}.000Z&ub={datetime_end.year}-{datetime_end.month:02d}-{datetime_end.day:02d}T{datetime_end.hour:02d}%3A{datetime_end.minute:02d}%3A{datetime_end.second:02d}.000Z"

    url = "https://www.gin5.admin.ch/gin/machine-api/api/value/station/prediction/"+location_id+"/"+product_id+"/"+"run"+string_date
    
    try:
        response_pred = requests.get(url, auth=HTTPBasicAuth(st.session_state['username'],st.session_state['password']))
        json_pred = response_pred.json()
        predictionRunId = [run['predictionRunId'] for run in json_pred.get('runs', [])]

        if predictionRunId is None or not predictionRunId:
            print('No runs found')
            return None
        else:
            for curr_id in predictionRunId:
                curr_url = "https://www.gin5.admin.ch/gin/machine-api/api/value/station/predictionrun/" + curr_id +"/result"
                response_pred = requests.get(curr_url, auth=HTTPBasicAuth(st.session_state['username'],st.session_state['password']))
                json_pred = response_pred.json()

                df_pred = pd.DataFrame(response_pred.json())
                df_pred['time'] = pd.to_datetime(df_pred['time'], utc=True)
                df_pred['time'] = df_pred['time'].dt.tz_convert(None)

                members_df = df_pred['members'].apply(pd.Series)
                members_df['time'] = df_pred['time']
                # Set time as index for plotting
                members_df.set_index('time', inplace=True)

                # if len(predictionRunId)>1:
                if predictionRunId.index(curr_id)==0:
                    df = {}
                ind_dict = members_df.index[0].strftime('%Y%m%d_%H%M')
                df[ind_dict]=members_df
                #else:
                    #df = members_df
        
    except ValueError:
        print(url)
        print(response_pred.text)
        return None

    return df


def update_data_obs_nonetcdf(measurements,date_end=None,date_start=None,ds =None):

    if date_end is None:
        date_end = pd.Timestamp.now()
    
    for id_var,name_var,units in zip(measurements[0]["id_variables"],measurements[0]["name_variables"],measurements[0]["variable_units"]):
        print('VARIABLE: ',name_var)
        if date_start is None:
            last_time = pd.to_datetime("2021-01-01")
        else:
            last_time = pd.to_datetime(date_start)
        
        if ds is not None:
            # Load existing file
            last_time = pd.to_datetime(ds["time"][-1].values)
            freq_obs = ds.attrs['freq_obs']
            existing_ds = True
        else:
            if 'ds' in locals():
                del ds
            freq_obs = list(np.zeros((len(measurements[0]['id_locations']))))
            existing_ds = False
        
        if last_time >= date_end:
            print("Data already up to date.")
        else:
            initialise_adding_data = True
            master_time_index = None
            locations = measurements[0]['name_locations']
            
            for id_station, name_station in zip(measurements[0]['id_locations'],measurements[0]['name_locations']):
                print(name_station,measurements[0]['full_name_location'][measurements[0]['id_locations'].index(id_station)], f"{measurements[0]['id_locations'].index(id_station)+1} of {len(measurements[0]['id_locations'])}")
                # Fetch new data
                df_new = fetch_obs(id_station,id_var, last_time, date_end)
                
                if df_new is not None and not df_new.empty:
                    # Create a datetime index from the earliest to latest time in df_new
                    start_time = df_new.index.min()
                    end_time = df_new.index.max()
                    # Assume fixed frequency (e.g. 5 minutes) – adjust as needed
                    inferred_freq = pd.infer_freq(df_new.index)
                    if inferred_freq is None:
                        if int((df_new.index[1]-df_new.index[0]).total_seconds() / 60) == int((df_new.index[-1]-df_new.index[-2]).total_seconds() / 60):
                            inferred_freq = f"{int((df_new.index[1]-df_new.index[0]).total_seconds() / 60)}min"
                        else:
                            print('issue with getting frequency: ',name_station)
                    
                    if existing_ds:
                        if freq_obs[measurements[0]['id_locations'].index(id_station)] == 0:
                            freq_obs[measurements[0]['id_locations'].index(id_station)]= inferred_freq
                        elif inferred_freq!=freq_obs[measurements[0]['id_locations'].index(id_station)]:
                            print('issue with frequency obs. Different from before!! Station: ',measurements[0]['name_locations'][measurements[0]['id_locations'].index(id_station)])
                    else:
                        freq_obs[measurements[0]['id_locations'].index(id_station)]=inferred_freq

                    print(inferred_freq)
                    if initialise_adding_data:
                        if inferred_freq is None:
                            inferred_freq = '5min'  # fallback if freq can't be inferred

                        master_time_index = pd.date_range(start=start_time, end=end_time, freq=inferred_freq)
                        data_storing = np.full((len(master_time_index), len(locations)), np.nan)
                        initialise_adding_data = False
                        print('Created data storage array:', data_storing.shape)
                    
                    # Reindex df_new to match master time index (fill missing with NaN)
                    df_aligned = df_new.reindex(master_time_index)
                    
                    # Insert data into correct column
                    col_idx = measurements[0]['id_locations'].index(id_station)
                    data_storing[:, col_idx] = df_aligned.to_numpy().flatten()
            
        if 'data_storing' in locals():
            # After loop, you can wrap data_storing and time into a DataFrame if needed
            df_all = pd.DataFrame(data_storing, index=master_time_index, columns=locations)
            ds_new = xr.Dataset(
                    data_vars=dict(
                        runoff=(["time", "location"], df_all.to_numpy())
                    ),
                    coords=dict(
                        time=pd.to_datetime(df_all.index).to_pydatetime(),
                        location=("location", df_all.columns)
                    )
                )
            
            if 'ds' in locals():
                ds_combined = xr.concat([ds, ds_new], dim="time")
                ds.close()
                ds_combined = ds_combined.sortby("time")

            else:
                ds_combined = ds_new
                ds_combined.attrs['time_zone']='UTC'
                ds_combined.attrs['units']=units
                ds_combined.attrs['variable']=name_var
                ds_combined.attrs['freq_obs']=freq_obs

            return ds_combined

# 1. Load data (cache it to avoid reloading on every interaction)
# @st.cache_data
def load_data():
    # if glob.glob(f"{PATH_STORING}/Runoff_2025.nc") == []:
    #     print('file does not exist')
    if 'obs' in st.session_state:
        print('obs in state, reloading')
        ds_old = st.session_state['obs']
        print(ds_old)
        if pd.to_datetime(ds_old["time"][-1].values)<now_:
            ds_new = update_data_obs_nonetcdf(measurements,date_end=now_+timedelta(hours=5*24),date_start=now_-timedelta(hours=5*24),ds=ds_old)
            ds = xr.concat([ds_old, ds_new], dim="time")
            ds = ds.sortby("time")
        else:
            ds = ds_old
        st.session_state['obs'] = ds
    else:
        print('Reading fresh')
        ds = update_data_obs_nonetcdf(measurements,date_end=now_+timedelta(hours=5*24),date_start=now_-timedelta(hours=5*24),ds=None)
    
    freq_obs = ds.attrs.get("freq_obs", [])
    locations = ds.location.values.tolist()

    stations__info = json.load(open('StationsIDs.json','r'))
    names_stations = [
    next((item["name"] for item in stations__info if item["shortName"] == st), None)
        for st in locations
        ]
    ds.attrs['locations']=locations
    ds.attrs['name_stations']=names_stations
    return ds

# 2. Placeholder function for "Update Data"
def update_data():
    if 'obs' in st.session_state:
        st.session_state['obs'] = update_data_obs_nonetcdf(measurements,date_end=now_,date_start=now_-timedelta(hours=5*24),ds=st.session_state['obs'])
    else:
        st.session_state['obs'] = update_data_obs_nonetcdf(measurements,date_end=now_,date_start=now_-timedelta(hours=5*24),ds=None)
    st.success(f"Data updated until {now_}")

measurements = [
    {
        "full_name_location": ["Rhein - Domat/Ems","Plessur - Chur","Rhein - Lustenau","Albula - Tiefencastel","Rhein - Bangs","Julia - Tiefencastel","Rhein - Wartau","Werdenb.Binnenkanal - Salez","Vorderrhein - Ilanz","Liechtenst. Binnenkan - Ruggell","Hinterrhein - Hinterrhein","Ill - Gisingen","Rhein - Maienfeld","Glenner - Castrisch","Landquart - Felsenbach","Rhein - Diepoldsau,Rietbrücke","Tamina - Bad Ragaz","Frutz - Sulz","Hinterrhein - Fürstenau"],
        "name_locations": ['B2602', 'B2185', 'VORARLBERG-200196', 'B2141', 'VORARLBERG-200014', 'B2418', 'SG-3304', 'B2187', 'B2033', 'B2410', 'B2631', 'VORARLBERG-200147', 'GR-4487', 'B2498', 'B2150', 'B2473', 'SG-3601', 'VORARLBERG-200642', 'B2387'],
        "id_locations": ['33910596-055b-41f1-8d93-9c790bac18bf', '188325cb-6d90-411a-ad02-33936878962a', '353edcc0-b9b5-4ea0-833c-ff46bcfd3f03', '25b8dd30-f445-4de2-86a6-c7f5ea5b2bd0', '592b5ff5-3409-4226-92fb-daf9fb9e81a1', '7bd36ff2-d46b-4d0b-a37f-47420ed48494', 'd41f9c93-7e74-4ac4-a28d-9cfbe710217c', '134674b5-ef5c-4898-82a0-f0a2eed19801', '50c08e8e-e9aa-4dab-951b-d796c6f7eb79', 'df304aab-85fe-4c51-a56f-f6b7ca2d9031', 'bf9f704f-58cb-40df-b884-d06567d1aece', 'dc41582b-4f95-4fea-88d3-e4748cb7e3b0', 'ea1a1444-e8f5-4aa4-8847-13fafce2d321', '7c486fb8-15a2-4771-8465-3c5f6e2d7f99', '89e530c7-cc79-4052-b272-774b1eacd7e0', '92abd866-c2a4-4f02-855d-022ee3e5e98b', '2d42b5c5-bfd7-4003-9559-4434dac77140', '3542bed6-4618-4726-a5e2-8ba6d7831d19', '741601c2-f0ad-44d8-aed7-f77323d92522'],
        "id_variables": ["6de7b8f1-cc60-40aa-b919-7c884856d89b"],
        "name_variables": ["Runoff"],
        "runoff_thresholds": [[780,1150,1800,2250],[],[],[80,110,120,130],[],[40,100,170,280],[],[65,90,100,110],[300,520,640,770],[20,40,50,70],[75,100,110,120],[],[],[130,260,350,460],[170,280,350,440],[1300,1950,2450,3050],[],[],[360,610,750,890]],
        "variable_units": ["m3/s"]
    }
]

products_ids_nwp_pred = ["354da43c-9106-484f-abe3-c842ca7a082a", "0411ead6-7ca6-4f82-9dc6-8fc29a0b1033", "6a40a112-64bc-4852-9076-208aa14bf3d9"]
names_nwp_pred = ['ICON-CH1-EPS','ICON-CH2-EPS','IFS']
chosen_start_times = [[1,3,5,9,11,15,17,21],[3,9,15,21],[7,19]] # First wasim run with each NWP model run 
# chosen_start_times = [[1,3,7,9,13,15,19,21],[5,13,17,21],[5,17]] # Last wasim run with each NWP model run 

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login Required")
    st.subheader('Please insert your username and password to access GIN API.')

    with st.form("login_form"):
        st.session_state['username'] = st.text_input("Username")
        st.session_state['password'] = st.text_input("Password", type="password")
        if 'wrong_credentials' in st.session_state and st.session_state['wrong_credentials']==True:
            st.markdown("Unfortunately the inserted username and password don't have access to GIN API. Please insert a valid account")
        submit = st.form_submit_button("Login")
        
        if submit:
            url = 'https://www.gin5.admin.ch/gin/machine-api/api/station?withroles=false'
            response_test = requests.get(url, auth=HTTPBasicAuth(st.session_state['username'],st.session_state['password']))
            
            if response_test.status_code == 200:
                st.session_state.logged_in = True
                st.session_state['wrong_credentials']=False
            else:
                st.session_state['wrong_credentials']=True
            
            st.rerun()

    st.stop()  # Stop app here if not logged in


st.sidebar.title("Select Date and Time")

# Option to choose 'Now' or 'Custom'
time_choice = st.sidebar.radio("Use current time or custom?", ["Now", "Custom"])

if 'now_' not in st.session_state:
    if time_choice == "Now":
        now_ = pd.Timestamp.now()
    else:
        # Define limits
        min_date = date(2022, 1, 1)
        max_date = datetime.now().date()

        selected_date = st.sidebar.date_input(
            "Select a date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

        selected_time = st.sidebar.time_input(
            "Select a time",
            value=datetime.now().time()
        )

        now_ = datetime.combine(selected_date, selected_time)

    st.sidebar.write("Selected datetime:\n", now_)

    start_time = now_-timedelta(hours=5*24)
    end_time   = now_

    proceed = st.sidebar.button("Continue")
    if proceed:
        st.session_state['now_']=now_

else:
    if time_choice == "Now":
        now_ = pd.Timestamp.now()
    else:
        # Define limits
        min_date = date(2022, 1, 1)
        max_date = datetime.now().date()

        selected_date = st.sidebar.date_input(
            "Select a date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

        selected_time = st.sidebar.time_input(
            "Select a time",
            value=datetime.now().time()
        )

        now_ = datetime.combine(selected_date, selected_time)

    st.sidebar.write("Selected datetime:\n", now_)

    start_time = now_-timedelta(hours=5*24)
    end_time   = now_

    proceed = st.sidebar.button("Update Date/Time")
    if proceed:
        st.session_state['now_']=now_
        st.session_state['initialise']=False
        del st.session_state['obs']


if 'now_' in st.session_state:
    if 'initialise' not in st.session_state or st.session_state['initialise'] is False:
        for nwps in names_nwp_pred:
            st.session_state[nwps] = {}
            for station in measurements[0]['full_name_location']:
                st.session_state[nwps][station]=None
            st.session_state['initialise']=True
        
    # 3. Main App
    placeholder_title = st.empty()
    placeholder_title.title("Runoff Observation Viewer:")
    placeholder_ = st.empty()

    # Sidebar
    st.sidebar.header("Controls")
    if 'obs' not in st.session_state:
        ds = load_data()
    else:
        ds = st.session_state['obs']

    st.session_state['obs']=ds


    freq_obs= ds.attrs["freq_obs"]
    location_list = ds.attrs["locations"]
    names_stations = ds.attrs["name_stations"]


    name_chosen = st.sidebar.selectbox("Choose a station", names_stations)
    placeholder_title.title(f"Runoff Observation Viewer: {name_chosen}")


    chosen_location = location_list[names_stations.index(name_chosen)]
    update_triggered = st.sidebar.button("Update Data", on_click=update_data)

    NWP_model = st.sidebar.selectbox("Predictions: ", ['None'] + names_nwp_pred)

    if NWP_model !='None':
        placeholder_title.title(f"Runoff Observation and Forecast Viewer: {name_chosen}")
        pred = None
        if NWP_model == 'ICON-CH1-EPS':
            hours_max_back = 24*2
        else:
            hours_max_back = 24*4
        # hours_back = 1

        # while pred is None and hours_back<48:
        #     print('Going back ', hours_back,'hours')
        if st.session_state[NWP_model][name_chosen] is None:
            pred = fetch_pred(measurements[0]['id_locations'][measurements[0]['full_name_location'].index(name_chosen)], products_ids_nwp_pred[names_nwp_pred.index(NWP_model)], now_-timedelta(hours=hours_max_back), now_)
            st.session_state[NWP_model][name_chosen] = pred
            
        else:
            pred = st.session_state[NWP_model][name_chosen]
        
        model_starts = list(pred.keys())
        model_starts_timestamps = sorted([pd.to_datetime(s, format='%Y%m%d_%H%M') for s in model_starts])
        model_starts = sorted(model_starts, key=lambda s: pd.to_datetime(s, format='%Y%m%d_%H%M'))

        model_starts_choose = []
        for k in model_starts:
            model_starts_choose.append(f"{k[6:8]}/{k[4:6]}/{k[0:4]} {k[9:11]}:{k[11:]}")
            # hours_back+=1
        if pred is None:
            NWP_model = None
            placeholder_.header(f'No predictions available in the last {hours_max_back}h at this station')

        end_time = pred[model_starts[-1]].index[-1]

    # Main panel: Load dataset again (without cache, for actual use)
    # ds = xr.open_dataset(f"{PATH_STORING}/Runoff_2025.nc", engine="netcdf4")

    # Extract series
    runoff_series = ds.runoff.sel(location=chosen_location)#.sel(time=slice(start_time, end_time))

    # Get frequency from attrs (assumes same order as locations)
    station_index = location_list.index(chosen_location)
    freq_str = freq_obs[station_index]
    freq = pd.to_timedelta(freq_str)

    # Filter out frequency-aligned NaNs
    if not runoff_series.isnull().all():
        first_valid_time = pd.to_datetime(str(runoff_series.dropna(dim="time", how="all").time.values[0]))
        expected_times = pd.date_range(start=first_valid_time, end=end_time, freq=freq)
        runoff_cleaned = runoff_series.sel(time=runoff_series.time.isin(expected_times))
    else:
        runoff_cleaned = runoff_series  # all NaNs


    # Plotting
    fig, ax = plt.subplots(figsize=(24, 8))
    # Obs
    runoff_cleaned.plot(ax=ax, color="black",label='obs.',zorder=100000)
    if NWP_model !='None':
        
            selection_timestamps = []
            selection_timestamps_show = []
            for timest_curr,k_curr in zip(model_starts_timestamps,model_starts):
                if timest_curr.hour in chosen_start_times[names_nwp_pred.index(NWP_model)]:
                    selection_timestamps.append(k_curr)
                    selection_timestamps_show.append(f"{k_curr[6:8]}/{k_curr[4:6]}/{k_curr[0:4]} {k_curr[9:11]}:{k_curr[11:]}")

            #adding latest run (if not already included)
            if model_starts[-1] not in selection_timestamps:
                k_curr = model_starts[-1]
                selection_timestamps.append(model_starts[-1])
                selection_timestamps_show.append(f"{k_curr[6:8]}/{k_curr[4:6]}/{k_curr[0:4]} {k_curr[9:11]}:{k_curr[11:]}")
            
            selection_timestamps_show=selection_timestamps_show[::-1]
            selection_timestamps=selection_timestamps[::-1]

            

            st.sidebar.header("WaSiM runs:")
            selected = st.sidebar.multiselect("Choose runs start time", options=selection_timestamps_show)

            # colors = [plt.colormaps['Blues_r'](i) for i in np.linspace(0.3, 0.9, len(selected))]
            colors = [plt.colormaps['copper'](i) for i in np.linspace(0.3, 0.9, len(selected))]



            for s in selected:
                
                id_plot = selected.index(s)
                id_run = selection_timestamps[selection_timestamps_show.index(s)]
                median_series = pred[id_run].median(axis=1)
                if NWP_model != 'IFS':
                    if selected.index(s)==0:
                        ax.plot(pred[id_run].index, median_series, label="Median ensembles", color="red", linewidth=2.5,zorder=(len(selected)-id_plot)*10)
                    else:
                        ax.plot(pred[id_run].index, median_series, label="_NONE", color="red", linewidth=2.5,zorder=(len(selected)-id_plot)*10)
                    # Pred
                add_label = True
                #Getting predictions
                
                for col in pred[id_run].columns:
                    if add_label:
                        label_ = s 
                        add_label=False
                    else:
                        label_ = '_none'
                    ax.plot(pred[id_run].index, pred[id_run][col], color=colors[id_plot],label=label_, linestyle="--",zorder=len(selected)-id_plot)

                # ax.plot(pred[id_run].index, median_series, label="_NONE", color="red", linewidth=2.5,zorder=id_plot)
                start_time = now_-timedelta(hours=max(hours_max_back,5*24))
        
        # elif choice_plotting == 'select one':
        #     choice_plotting_run = st.sidebar.selectbox("Forecasts time: ", ['None']+model_starts_choose)
        #     if choice_plotting_run != 'None':
        #         id_run = model_starts[model_starts_choose.index(choice_plotting_run)]
        #         # Pred
        #         add_label = True
        #         #Getting predictions
        #         for col in pred[id_run].columns:
        #             if add_label:
        #                 label_ = 'ensembles'
        #                 add_label=False
        #             else:
        #                 label_ = '_none'
        #             ax.plot(pred[id_run].index, pred[id_run][col], color="orange",label=label_, linestyle="--")

        #         median_series = pred[id_run].median(axis=1)
        #         ax.plot(pred[id_run].index, median_series, label="Median ensembles", color="red", linewidth=2.5)
        #         end_time = pred[id_run].index[-1]
        #         start_time = now_-timedelta(hours=hours_max_back)

    # Set larger fonts
    title_fontsize = 24
    label_fontsize = 20
    tick_fontsize = 16

    if end_time>now_:
        ax.axvline(now_,color="black", linestyle="--", linewidth=2, label="_Now")

    # print(measurements['full_name_location'].index(name_chosen))

    thresholds = measurements[0]['runoff_thresholds'][measurements[0]['full_name_location'].index(name_chosen)]

    if len(thresholds)>0:
        t0, t1, t2, t3 = thresholds
        ax.axhline(t0, color='yellow', linewidth=1.5)
        ymin, ymax = ax.get_ylim()
        # 3. Light yellow background between t0 and t1
        ax.axhspan(t0, t1, facecolor='yellow', alpha=0.2)

        # 4. Light orange background between t1 and t2
        ax.axhspan(t1, t2, facecolor='orange', alpha=0.2)

        # 5. Light red background between t2 and t3
        ax.axhspan(t2, t3, facecolor='red', alpha=0.2)

        # 6. Light brown background above t3 (up to current ymax)
        ax.axhspan(t3, ymax, facecolor='brown', alpha=0.2)

        # 7. Restore original limits so nothing shifts
        ax.set_ylim(ymin, ymax)

    ax.set_title(f"Runoff at {chosen_location}", fontsize=title_fontsize)
    ax.set_xlabel(" ", fontsize=label_fontsize)
    ax.xaxis.get_offset_text().set_fontsize(tick_fontsize)
    ax.set_ylabel("Runoff (m³/s)", fontsize=label_fontsize)
    ax.legend(fontsize=label_fontsize)

    # Set tick label sizes
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    ax.set_xlim([start_time, end_time])
    ax.grid(True)
    st.pyplot(fig)

    ds.close()

# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import math 
import os 

inputs_path = os.path.join(os.getcwd(), 'inputs')

output_path = 'outputs'
output_name = 'superwell_python_test'  # file name for output
output_file = os.path.join(output_path, f'{output_name}.csv')

# load data 
grid_df = pd.read_csv(os.path.join(inputs_path, 'inputs.csv'), keep_default_na=False)
well_params = pd.read_csv(os.path.join(inputs_path, 'Well_Params.csv'), index_col=0)
electricity_rates = pd.read_csv(os.path.join(inputs_path, 'GCAM_Electrical_Rates.csv'), index_col='country')
W_lookup = pd.read_csv(os.path.join(inputs_path, 'Theis_well_function_table.csv'))
lookup_idx = pd.Index(W_lookup.W)

# define constants
SECS_IN_DAY = 24 * 60 * 60

# user defined 
IRR_DEPTH = 0.30  # annual irrigation depth target (m)
NUM_YEARS = 100  # maximum years of pumping
DAYS = 100  # days pumping per year
MAXIMUM_INITIAL_SAT_THICKNESS = 200
NUM_INTERFERENCE_WELLS = 2
WELL_DEEPENING_DEPTH = 50  # m

# screening criteria
max_s_frac = 0.40  # max drawdown as % of sat thickness
max_s_absolute = 80  # max drawdown in m

# imported 
ELECTRICITY_RATE =  well_params.Val['Energy_cost_rate'] # defualt electricity rate
DEPLETION_LIMIT = well_params.Val['Depletion_Limit'] # depletion limit for this scenario 
EFFICIENCY = well_params.Val['Pump_Efficiency'] # well efficiency
WELL_LIFETIME = well_params.Val['Max_Lifetime_in_Years']
INTEREST_RATE = well_params.Val['Interest_Rate']
MAINTENANCE_RATE = well_params.Val['Maintenance_factor']
SPECIFIC_WEIGHT = well_params.Val['Specific_weight'] # specific weight of water 

# convert electricity rate dictionary 
electricity_rate_dict = electricity_rates.electricity_cost_dollar_per_KWh.to_dict()
    
# filter by country, if desired
country = 'United States'
if country != 'all':
    grid_df = grid_df[grid_df.CNTRY_NAME == country]

################ determine if grid cell is skipped ########################
grid_df = grid_df[
    (grid_df.Area >= 10**7) & # skip grid areas less than 10x10 km
    (grid_df.Depth >= 1) &  # depth to water table should we at least 5 meters
    (grid_df.Permeability >= -15) & # limit low permeability values
    (grid_df.Porosity >= 0.05) # limit porosity to 5% voids at least
]

# correct aquifer thickness outliers, replace >1000m thickness with 1000m
grid_df.Thickness.loc[grid_df.Thickness > 1000] = 1000

grid_df = grid_df.reset_index(drop=True)  # so that it starts at 0


# define Theis function
def drawdown_theis(time, r, S, T, Q):
    u = r**2 * S/(4 * T * time)

    if u < 0.6:  # use approximation for small u values
        W = -np.euler_gamma - math.log(u) + u - u**2/(2*2)
        
    elif 0.6 <= u <= 5.9:  # use W(u) lookup table for intermediate values where approximation is insufficient
        W = W_lookup.W[(W_lookup.u - u).abs().idxmin()]
        
    else:  # for large u values, W will be insignificant and drawdown (s) will ~= 0
        W = 0

    s = W * Q / (4 * np.pi * T)
    
    return s


def get_pumping_rate(time, r, S, T, Q_candidates):
    for Q in sorted(Q_candidates, reverse=True):
        s = drawdown_theis(time, r, S, T, Q)
        if s / initial_sat_thickness < max_s_frac and s < max_s_absolute:
            return Q
    return 0


def simulate_pumping(r, roi, S, T, Q, sat_thickness, days=100, timestep=10):
    s_theis_ts = []
    s_theis_interference_ts = []

    for day in range(timestep, days + 1, timestep):
        s_theis_ts.append(drawdown_theis(day * SECS_IN_DAY, r, S, T, Q))
        s_theis_interference_ts.append(drawdown_theis(day * SECS_IN_DAY, roi * 2, S, T, Q))

    # average drawdown
    s_interference_avg = 4 * np.mean(s_theis_interference_ts)
    s_theis_avg = np.mean(s_theis_ts) + s_interference_avg

    # convert to Jacob - solve quadratic
    a = -1 / (2 * sat_thickness)
    b = 1
    c = -s_theis_avg

    s_jacob = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
    # s_jacob = (1 - np.sqrt(1 - 2 * s_theis_avg / sat_thickness)) * sat_thickness  # ?
    #root_2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)

    return s_jacob, s_interference_avg

    
# candidate well pumping rates (gallons per minute)
Q_array_gpm = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1200, 1300, 1400, 1500]

Q_array = [i / (60 * 264.17) for i in Q_array_gpm]  # convert gpm to m^3/s

columns = ['DTW', 'sat_thickness', 'well_length', 'T', 'Well_Q', 'well_roi', 'well_area', 'depleted_volume_fraction',
           'drawdown', 'drawdown_interference', 'total_head', 'volume_per_well', 'num_wells', 'volume_all_wells',
           'cumulative_volume_per_well', 'cumulative_volume_all_wells', 'annual_capital_cost', 'maintenance_cost',
           'well_installation_cost', 'nonenergy_cost', 'power', 'energy', 'energy_cost_rate', 'energy_cost',
           'total_cost_per_well', 'total_cost_all_wells', 'unit_cost', 'unit_cost_per_km3', 'unit_cost_per_acreft']
pd.DataFrame(columns=columns).to_csv(output_file, mode='w')  # write headers to file

#%% superwell code block 
for grid_cell in grid_df.itertuples(name='GridCell'):

    # initialize an empty dataframe to store values for each year
    df = pd.DataFrame(data=0, index=range(NUM_YEARS), columns=columns)
    
    print('Percent complete = ' + str(100 * grid_cell.Index/len(grid_df)))  # TODO: progress bar package?

    ################ store grid cell attributes for output ####################
    
    ELECTRICITY_RATE = electricity_rate_dict[grid_cell.CNTRY_NAME]

    # depth to water 
    df.DTW[0] = grid_cell.Depth  # initial depth to water
    
    # saturated thickness 
    initial_sat_thickness = grid_cell.Thickness - grid_cell.Depth  # m

    if initial_sat_thickness > MAXIMUM_INITIAL_SAT_THICKNESS:
        df.sat_thickness[0] = MAXIMUM_INITIAL_SAT_THICKNESS  # m
        df.well_length[0] = df.sat_thickness[0] + df.DTW[0]  # m
    else:    
        df.sat_thickness[0] = initial_sat_thickness  # m
        df.well_length[0] = grid_cell.Thickness  # m
        
    # available volume
    available_volume = initial_sat_thickness * grid_cell.Area * grid_cell.Porosity
    
    # aquifer properties for Theis 
    S = grid_cell.Porosity  # [-]
    K = (10 ** grid_cell.Permeability) * 1e7  # m/s
    T = K * df.sat_thickness[0]  # m/s
    df['T'][0] = T  # initial T  # pandas interprets df.T as the transpose of df, so we need df['T'] to get column T
    
    # assign well unit cost based on WHY Class 
    if grid_cell.WHYClass == 10:
        well_unit_cost = well_params.Val['Well_Install_10']
    elif grid_cell.WHYClass == 20:
        well_unit_cost = well_params.Val['Well_Install_20']
    else:
        well_unit_cost = well_params.Val['Well_Install_30']
    
    #################### determine initial well Q #############################
    
    # time and well radius for Theis solution
    time_Q = 2 * 365 * SECS_IN_DAY  # time period used for determining initial well Q
    well_r = 0.5 * well_params.Val['Well_Diameter']
    
    # find largest Q that meets screening criteria
    df.Well_Q[0] = get_pumping_rate(time_Q, well_r, S, T, Q_array)

    # skip grid cell if no pumping rates are viable
    if df.Well_Q[0] == 0:
        continue

    ###################### determine initial well Area ########################
    df.well_area[0] = df.Well_Q[0] * DAYS * SECS_IN_DAY / IRR_DEPTH  # m^2
    df.well_roi[0] = np.sqrt(df.well_area[0] / np.pi)  # m

    ####################### annual pumping simulation loop ####################

    for year in range(NUM_YEARS):
        if df.depleted_volume_fraction[year-1] > DEPLETION_LIMIT:
            year = year - 1 
            break
            
        # test viability for current year (simulate drawdown at t = 100 days of pumping)
        s_theis = drawdown_theis(DAYS * SECS_IN_DAY, well_r, S, df['T'][year], df.Well_Q[year])
        s_theis_interference = drawdown_theis(DAYS * SECS_IN_DAY, df.well_roi[year] * NUM_INTERFERENCE_WELLS, S, df['T'][year], df.Well_Q[year])
        s_total = s_theis + 4 * s_theis_interference  # total drawdown (well + interference)

        # check if drawdown constraints are violated by end of 100 day pumping period
        # if constraints violated: (1) first deepen well, (2) then reduce well pumping rate 
        if s_total / df.sat_thickness[year] > max_s_frac or s_total > max_s_absolute:
            
            # 1) first preference deepen well 
            if df.well_length[year] < grid_cell.Thickness:
                # deepen well length by WELL_DEEPENING_DEPTH, but do not exceed aquifer thickness
                df.well_length[year] = min(df.well_length[year] + WELL_DEEPENING_DEPTH, grid_cell.Thickness)

                # update saturated thickness and T 
                df.sat_thickness[year] = df.well_length[year] - df.DTW[year]
                df['T'][year] = df.sat_thickness[year] * K
                
            # 2) once well cannot be deepened, reduce well pumping rate 
            else:
                # find largest Q that meets screening criteria
                df.Well_Q[year] = get_pumping_rate(time_Q, well_r, S, df['T'][year], Q_array)

                # exit pumping code block if no pumping rates are viable
                if df.Well_Q[year] == 0:
                    break

                # update roi
                df.well_area[year] = df.Well_Q[year] * DAYS * SECS_IN_DAY / IRR_DEPTH
                df.well_roi[year] = np.sqrt(df.well_area[year] / np.pi)
           
        # if constraints aren't violated, proceed to calculate output for pumping year  
        # simulate 100 days of pumping, with drawdown calculated every 10 days
        s_jacob, s_interference_avg = simulate_pumping(well_r, df.well_roi[year], S, T, df.Well_Q[year], df.sat_thickness[year])

        ########################### compute outputs ###########################
        
        # save annual pumping values to arrays
        df.drawdown[year] = s_jacob
        df.drawdown_interference[year] = s_interference_avg
        df.total_head[year] = s_jacob + df.DTW[year]
        df.volume_per_well[year] = df.Well_Q[year] * SECS_IN_DAY * DAYS
        df.num_wells[year] = grid_cell.Area / df.well_area[year]
        df.volume_all_wells[year] = df.volume_per_well[year] * df.num_wells[year]
        df.cumulative_volume_per_well[year] = df.volume_per_well[year] + df.cumulative_volume_per_well[year-1]
        df.cumulative_volume_all_wells[year] = df.volume_all_wells[year] + df.cumulative_volume_all_wells[year-1]
        df.depleted_volume_fraction[year] = df.cumulative_volume_all_wells[year] / available_volume

        # update variable arrays for next annual pumping iteration
        if year != NUM_YEARS-1:
            df.Well_Q[year+1] = df.Well_Q[year]
            df.DTW[year+1] = df.DTW[year] + (df.volume_all_wells[year]/grid_cell.Area)/S
            df.sat_thickness[year+1] = df.well_length[year] - df.DTW[year+1]
            df['T'][year+1] = K * df.sat_thickness[year+1]
            df.well_roi[year+1] = df.well_roi[year]
            df.well_area[year+1] = df.well_area[year]
            df.well_length[year+1] = df.well_length[year]
            
    ##################### annual costs and unit costs ######################### 
   
    # find indexes of years when number of wells increase due to pumping rate reduction 
    # along with pumping rate and corresponding number of wells
    pumping_years = year + 1
    well_count = np.unique(df.num_wells)
    if min(well_count) == 0:
        well_count = np.delete(well_count, 0)
        
    added_well_count = np.zeros(len(well_count))
    for i in range(len(added_well_count)):
        if i == 0:
            added_well_count[i] = well_count[i]
        else:
            added_well_count[i] = well_count[i] - well_count[i-1]
    
    Q_vals = np.unique(df.Well_Q)
    if min(Q_vals) == 0:
        Q_vals = np.delete(Q_vals, 0)
    Q_vals = np.sort(Q_vals)    
    Q_vals = Q_vals[::-1]
        
    Start_indx = np.zeros(len(Q_vals)) # indexes where pumping rate and well num changes 
    if len(Start_indx) == 1:
        pass
    
    else:
        for i in range(pumping_years):
            if i == 0:
                counter = 1
                continue 
            if df.num_wells[i] - df.num_wells[i-1] > 0:
                Start_indx[counter] = int(i)
                counter += 1 
    
    # initialize cost arrays to track annual non-energy costs for each group of added wells 
    capital_cost_array = np.zeros((len(Start_indx), int(NUM_YEARS + WELL_LIFETIME)))
    maintenance_array = np.zeros((len(Start_indx), int(NUM_YEARS + WELL_LIFETIME)))
            
    # 1) no deepening, initial_sat_thickness < MAXIMUM_INITIAL_SAT_THICKNESS
    if initial_sat_thickness < MAXIMUM_INITIAL_SAT_THICKNESS:
        install_cost = well_unit_cost * df.well_length[0] # if no deepening, well install remains fixed 
        for added_wells in range(len(added_well_count)):
            offset = int(Start_indx[added_wells]) 
            for year in range(pumping_years):
                capital_cost_array[added_wells, year + offset] = added_well_count[added_wells] * install_cost * ((1 + INTEREST_RATE) ** WELL_LIFETIME) * INTEREST_RATE/((1 + INTEREST_RATE) ** WELL_LIFETIME-1)
                maintenance_array[added_wells, year + offset] = MAINTENANCE_RATE * install_cost * added_well_count[added_wells] # maintenance cost [% of initial cost]

    # 2) deepening, initial_sat_thickness > MAXIMUM_INITIAL_SAT_THICKNESS
    else:
        for added_wells in range(len(added_well_count)):
            offset = int(Start_indx[added_wells]) 
            for year in range(pumping_years):
                if year + offset == pumping_years:
                    break
                
                elif year == 0: 
                    install_cost = well_unit_cost * df.well_length[0] 
                    capital_cost_array[added_wells, year + offset] = added_well_count[added_wells] * install_cost * ((1 + INTEREST_RATE) ** WELL_LIFETIME) * INTEREST_RATE/((1 + INTEREST_RATE) ** WELL_LIFETIME-1)
                    maintenance_array[added_wells, year + offset] = MAINTENANCE_RATE * install_cost * added_well_count[added_wells] # maintenance cost [% of initial cost]
                    
                elif (year+1)/WELL_LIFETIME == 0: # Replace well every n years (well lifetime), if reduced yeild, cheaper unit cost at 200 gpm and below
                        
                    install_cost = well_unit_cost * df.well_length[year + offset]
                    capital_cost_array[added_wells, year + offset] += added_well_count[added_wells] * install_cost * ((1 + INTEREST_RATE) ** WELL_LIFETIME) * INTEREST_RATE/((1 + INTEREST_RATE) ** WELL_LIFETIME-1)
                    maintenance_array[added_wells, year + offset] += MAINTENANCE_RATE * install_cost * added_well_count[added_wells] # maintenance cost [% of initial cost]
            
                elif df.well_length[year + offset] - df.well_length[year - 1 + offset] > 0:
                    capital_cost_array[added_wells, year + offset] += added_well_count[added_wells] * install_cost * ((1 + INTEREST_RATE) ** WELL_LIFETIME) * INTEREST_RATE/((1 + INTEREST_RATE) ** WELL_LIFETIME-1)
                    capital_cost_array[added_wells, (year + offset): int((year + offset + WELL_LIFETIME))] += well_unit_cost * (df.well_length[year + offset] - df.well_length[year - 1 + offset]) * ((1 + INTEREST_RATE) ** WELL_LIFETIME) * INTEREST_RATE/((1 + INTEREST_RATE) ** WELL_LIFETIME-1) * added_well_count[added_wells]
                    install_cost = well_unit_cost * df.well_length[year + offset]
                    maintenance_array[added_wells, year + offset] += MAINTENANCE_RATE * install_cost * added_well_count[added_wells]
                    
                else:
                    capital_cost_array[added_wells, year + offset] += added_well_count[added_wells] * install_cost * ((1 + INTEREST_RATE) ** WELL_LIFETIME) * INTEREST_RATE/((1 + INTEREST_RATE) ** WELL_LIFETIME-1)
                    maintenance_array[added_wells, year + offset] += MAINTENANCE_RATE * install_cost * added_well_count[added_wells] # maintenance cost [% of initial cost]
                    
    ####################### annual cost metrics ###########################

    df['annual_capital_cost'] = np.sum(capital_cost_array, axis=0)
    df['maintenance_cost'] = np.sum(maintenance_array, axis=0)

    # compute derived columns
    df.well_installation_cost = well_unit_cost * df.well_length
    df.nonenergy_cost = df.annual_capital_cost + df.maintenance_cost
    df.power = df.num_wells * (SPECIFIC_WEIGHT * df.total_head * df.Well_Q / EFFICIENCY) / 1000  # kW
    df.energy = df.power * (DAYS * 24)  # kWh/year
    df.energy_cost_rate = ELECTRICITY_RATE  # $ per kWh
    df.energy_cost = df.energy * df.energy_cost_rate  # $/year
    df.total_cost_per_well = (df.nonenergy_cost + df.energy_cost) / df.num_wells
    df.total_cost_all_wells = df.num_wells * df.total_cost_per_well
    df.unit_cost = df.total_cost_all_wells / df.volume_all_wells  # $/m^3
    df.unit_cost_per_km3 = df.unit_cost * 10 ** 9  # $/km^3
    df.unit_cost_per_acreft = df.unit_cost * 1233.48  # $/acft

    df[:pumping_years].to_csv(output_file, mode='a', header=False)  # append to file

    ######################## save grid cell results ###########################
    """
    ['iteration', 'year_number', 'DEPLETION_LIMIT', 'continent', 'country_name', 
    'gcam_basin_id', 'gcam_basin_name', 'well_id', 'grid_area', 'permeability', 'storativity', 
    'total_thickness', 'depth_to_piez_surface', 'orig_aqfr_sat_thickness', 'aqfr_sat_thickness', 
    'hydraulic_conductivity', 'transmissivity', 'radius_of_influence', 'areal_extent', 
    'max_drawdown', 'drawdown', 'drawdown_interference', 'total_head', 'well_yield', 'volume_produced_perwell', 
    'cumulative_vol_produced_perwell', 'number_of_wells', 'volume_produced_allwells', 
    'cumulative_vol_produced_allwells', 'available_volume', 'depleted_vol_fraction', 
    'well_installation_cost', 'annual_capital_cost', 'maintenance_cost', 'nonenergy_cost', 
    'power', 'energy', 'energy_cost_rate', 'energy_cost', 'total_cost_perwell', 'total_cost_allwells', 
    'unit_cost', 'unit_cost_per_km3', 'unit_cost_per_acreft', 'whyclass', 'total_well_length']
    """

#%% Plot distributions of results 

os.chdir('outputs')
results = pd.read_csv('superwell_python_output.csv')

first_year_unit_cost = results.where(results.year_number == 1)
first_year_unit_cost = first_year_unit_cost.dropna(thresh = 5)

import matplotlib.pyplot as plt 

plt.scatter(first_year_unit_cost.hydraulic_conductivity * SECS_IN_DAY, first_year_unit_cost.unit_cost_per_acreft)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Aquifer K (m/d)')
plt.ylabel('$ per acre-foot')

plt.scatter(first_year_unit_cost.well_yield * 60 * 264.17, first_year_unit_cost.unit_cost_per_acreft)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Well pumping rate (gpm)')
plt.ylabel('$ per acre-foot')

plt.scatter(first_year_unit_cost.areal_extent * 0.000247105, first_year_unit_cost.unit_cost_per_acreft)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Acres per well')
plt.ylabel('$ per acre-foot')

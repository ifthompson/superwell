# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import math 
import os 

inputs_path = os.path.join(os.getcwd(), 'inputs')

output_path = 'outputs'
output_name = 'superwell_python_test'  # file name for output

# load data 
grid_df = pd.read_csv(os.path.join(inputs_path, 'inputs.csv'), keep_default_na=False)
well_params = pd.read_csv(os.path.join(inputs_path, 'Well_Params.csv'), index_col=0)
electricity_rates = pd.read_csv(os.path.join(inputs_path, 'GCAM_Electrical_Rates.csv'), index_col='country')
W_lookup = pd.read_csv(os.path.join(inputs_path, 'Theis_well_function_table.csv'), header="infer")
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
        lookup_idx = pd.Index(W_lookup.u)
        lookup_loc = lookup_idx.get_loc(u, method='nearest')
        W = W_lookup.W[lookup_loc]
        
    else:  # for large u values, W will be insignificant and drawdown (s) will ~= 0
        W = 0

    s = W * Q / (4 * np.pi * T)
    
    return s
    
# candidate well pumping rates (gallons per minute)
Q_array_gpm = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1200, 1300, 1400, 1500]

# TODO: intervals?

Q_array = [i / (60 * 264.17) for i in Q_array_gpm]  # convert gpm to m^3/s

# TODO: Hassan pick important variables, get rid of constants
# header for output file 
header_column_names = 'iteration,year_number,DEPLETION_LIMIT,continent,country_name,' \
        'gcam_basin_id,gcam_basin_name,well_id,grid_area,permeability,storativity,' \
        'total_thickness,depth_to_piez_surface,orig_aqfr_sat_thickness,aqfr_sat_thickness,' \
        'hydraulic_conductivity,transmissivity,radius_of_influence,areal_extent,' \
        'max_drawdown,drawdown,drawdown_interference,total_head,well_yield,volume_produced_perwell,' \
        'cumulative_vol_produced_perwell,number_of_wells,volume_produced_allwells,' \
        'cumulative_vol_produced_allwells,available_volume,depleted_vol_fraction,' \
        'well_installation_cost, annual_capital_cost,maintenance_cost,nonenergy_cost,' \
        'power,energy,energy_cost_rate,energy_cost,total_cost_perwell,total_cost_allwells,' \
        'unit_cost,unit_cost_per_km3,unit_cost_per_acreft,whyclass,total_well_length'

outputs_df = pd.DataFrame(columns=['well_id', 'year', 'total_thickness', 'depth_to_piez_surface', 'aqfr_sat_thickness',
                                   'radius_of_influence', 'drawdown', 'drawdown_interference', 'total_head',
                                   'well_yield', 'volume_produced_perwell', 'number_of_wells', 'depleted_vol_fraction'])

#%% superwell code block 
for grid_cell in grid_df.itertuples(name='GridCell'):
    
    print('Percent complete = ' + str(100 * grid_cell.Index/len(grid_df)))  # TODO: progress bar package?

    # initialize all yearly variables containers
    DTW_array = np.zeros(NUM_YEARS)  # tracks depth to water for each year
    sat_thickness_array = np.zeros(NUM_YEARS)
    well_length_array = np.zeros(NUM_YEARS)
    T_array = np.zeros(NUM_YEARS)  # tracks T for each year
    Well_Q_array = np.zeros(NUM_YEARS)
    well_roi_array = np.zeros(NUM_YEARS)
    well_area_array = np.zeros(NUM_YEARS)
    depleted_volume_fraction = np.zeros(NUM_YEARS)  # initialize
    drawdown = np.zeros(NUM_YEARS)
    drawdown_interference = np.zeros(NUM_YEARS)
    total_head = np.zeros(NUM_YEARS)
    volume_per_well = np.zeros(NUM_YEARS)
    num_wells = np.zeros(NUM_YEARS)
    volume_all_wells = np.zeros(NUM_YEARS)
    cumulative_volume_per_well = np.zeros(NUM_YEARS)
    cumulative_volume_all_wells = np.zeros(NUM_YEARS)

    annual_capital_cost = np.zeros(NUM_YEARS)
    maintenance_cost = np.zeros(NUM_YEARS)
    well_installation_cost = np.zeros(NUM_YEARS)
    nonenergy_cost = np.zeros(NUM_YEARS)
    power = np.zeros(NUM_YEARS)
    energy = np.zeros(NUM_YEARS)
    energy_cost_rate = np.zeros(NUM_YEARS)
    energy_cost = np.zeros(NUM_YEARS)
    total_cost_per_well = np.zeros(NUM_YEARS)
    total_cost_all_wells = np.zeros(NUM_YEARS)
    unit_cost = np.zeros(NUM_YEARS)
    unit_cost_per_km3 = np.zeros(NUM_YEARS)
    unit_cost_per_acreft = np.zeros(NUM_YEARS)

    ################ store grid cell attributes for output ####################
    
    ELECTRICITY_RATE = electricity_rate_dict[grid_cell.CNTRY_NAME]

    # depth to water 
    DTW_array[0] = grid_cell.Depth  # initial depth to water
    
    # saturated thickness 
    initial_sat_thickness = grid_cell.Thickness - grid_cell.Depth  # m

    if initial_sat_thickness > MAXIMUM_INITIAL_SAT_THICKNESS:
        sat_thickness_array[0] = MAXIMUM_INITIAL_SAT_THICKNESS  # m
        well_length_array[0] = sat_thickness_array[0] + DTW_array[0]  # m
    else:    
        sat_thickness_array[0] = initial_sat_thickness  # m
        well_length_array[0] = grid_cell.Thickness  # m
        
    # available volume
    available_volume = initial_sat_thickness * grid_cell.Area * grid_cell.Porosity
    
    # aquifer properties for Theis 
    S = grid_cell.Porosity  # [-]
    K = (10 ** grid_cell.Permeability) * 1e7  # m/s
    T = K * sat_thickness_array[0]  # m/s
    T_array[0] = T  # initial T
    
    # assign well unit cost based on WHY Class 
    if grid_cell.WHYClass == 10:
        well_unit_cost = well_params.Val['Well_Install_10']
    elif grid_cell.WHYClass == 20:
        well_unit_cost = well_params.Val['Well_Install_20']
    else:
        well_unit_cost = well_params.Val['Well_Install_30']
    
    #################### determine initial well Q #############################
    
    # time and well radius for Theis solution
    # TODO: why times 2?
    time_Q = 2 * 365 * SECS_IN_DAY  # time period used for determining initial well Q
    well_r = 0.5 * well_params.Val['Well_Diameter']
    
    # find largest Q that meets screening criteria
    initial_Q = None
    for Q in reversed(Q_array):
        s = drawdown_theis(time_Q, well_r, S, T, Q)  # drawdown at t = 2 years for candidate well
        if s/initial_sat_thickness < max_s_frac and s < max_s_absolute:
            initial_Q = Q
            break

    # skip grid cell if no pumping rates are viable
    if initial_Q is None:
        continue

    Well_Q_array[0] = initial_Q
    
    ###################### determine initial well Area ########################
    initial_well_area = initial_Q * DAYS * SECS_IN_DAY / IRR_DEPTH  # m^2
    initial_roi = np.sqrt(initial_well_area / np.pi)  # m
    well_roi_array[0] = initial_roi
    well_area_array[0] = initial_well_area

    ####################### annual pumping simulation loop ####################

    # TODO: restructure with grid cell class an run year method
    for year in range(NUM_YEARS):
        if depleted_volume_fraction[year-1] > DEPLETION_LIMIT:
            year = year - 1 
            break
            
        # test viability for current year (simulate drawdown at t = 100 days of pumping)
        s_theis = drawdown_theis(DAYS * SECS_IN_DAY, well_r, S, T_array[year], Well_Q_array[year])
        s_theis_interference = drawdown_theis(DAYS * SECS_IN_DAY, well_roi_array[year] * NUM_INTERFERENCE_WELLS, S, T_array[year], Well_Q_array[year])
        s_total = s_theis + 4 * s_theis_interference  # total drawdown (well + interference)
        # TODO: question, relationship between NUM_INTERFERENCE_WELLS and "4"
        
        # check if drawdown constraints are violated by end of 100 day pumping period
        # if constraints violated: (1) first deepen well, (2) then reduce well pumping rate 
        if s_total/sat_thickness_array[year] > max_s_frac or s_total > max_s_absolute:
            
            # 1) first preference deepen well 
            if well_length_array[year] < grid_cell.Thickness:
                # deepen well length by WELL_DEEPENING_DEPTH, but do not exceed aquifer thickness
                well_length_array[year] = min(well_length_array[year] + WELL_DEEPENING_DEPTH, grid_cell.Thickness)

                # update saturated thickness and T 
                sat_thickness_array[year] = well_length_array[year] - DTW_array[year]
                T_array[year] = sat_thickness_array[year] * K
                
            # 2) once well cannot be deepened, reduce well pumping rate 
            else:

                # TODO: put in function for reuse
                # find largest Q that meets screening criteria
                new_Q = None
                for Q in reversed(Q_array):
                    s = drawdown_theis(time_Q, well_r, S, T_array[year], Q)  # drawdown at t = 2 years for candidate well
                    if s / initial_sat_thickness < max_s_frac and s < max_s_absolute:
                        new_Q = Q
                        break

                # exit pumping code block if no pumping rates are viable
                if new_Q is None:
                    break

                Well_Q_array[year] = new_Q  # update Q for current YEAR
                
                # update roi
                well_area_array[year] = Well_Q_array[year] * DAYS * SECS_IN_DAY / IRR_DEPTH
                well_roi = np.sqrt(well_area_array[year] / np.pi)
                well_roi_array[year] = well_roi  # TODO: this was initial_roi, was it a bug?
           
        # if constraints aren't violated, proceed to calculate output for pumping year  
        # simulate 100 days of pumping, with drawdown calculated every 10 days
        s_theis_ts = np.zeros(int(DAYS/10)) 
        s_theis_interference_ts = np.zeros(int(DAYS/10))

        for day in range(int(DAYS/10)):
            s_theis_ts[day] = drawdown_theis((day+1) * 10 * SECS_IN_DAY, well_r, S, T_array[year], Well_Q_array[year])
            s_theis_interference_ts[day] = drawdown_theis((day+1) * 10 * SECS_IN_DAY, well_roi_array[year] * 2, S, T_array[year], Well_Q_array[year])
        
        # average drawdown
        s_theis_avg = np.mean(s_theis_ts) + np.mean(4 * s_theis_interference_ts)
        s_interference_avg = 4 * np.mean(s_theis_interference_ts)
        
        # convert to Jacob - solve quadratic
        a = -1/(2*sat_thickness_array[year])
        b = 1
        c = -s_theis_avg
        
        root_1 = (-b + (b**2 - 4 * a * c) ** 0.5)/(2*a) 
        root_2 = (-b - (b**2 - 4 * a * c) ** 0.5)/(2*a) 
        
        s_jacob = root_1
        
        ########################### compute outputs ###########################
        
        # save annual pumping values to arrays
        drawdown[year] = s_jacob
        drawdown_interference[year] = s_interference_avg
        total_head[year] = s_jacob + DTW_array[year]
        volume_per_well[year] = Well_Q_array[year] * SECS_IN_DAY * DAYS
        num_wells[year] = grid_cell.Area/well_area_array[year]
        volume_all_wells[year] = volume_per_well[year] * num_wells[year]
        cumulative_volume_per_well[year] = volume_per_well[year] + cumulative_volume_per_well[year-1]
        cumulative_volume_all_wells[year] = volume_all_wells[year] + cumulative_volume_all_wells[year-1]
        depleted_volume_fraction[year] = cumulative_volume_all_wells[year]/available_volume

        # update variable arrays for next annual pumping iteration
        if year != NUM_YEARS-1:
            Well_Q_array[year+1] = Well_Q_array[year]
            DTW_array[year+1] = DTW_array[year] + (volume_all_wells[year]/grid_cell.Area)/S
            sat_thickness_array[year+1] = well_length_array[year] - DTW_array[year+1]
            T_array[year+1] = K * sat_thickness_array[year+1]
            well_roi_array[year+1] = well_roi_array[year]
            well_area_array[year+1] = well_area_array[year]
            well_length_array[year+1] = well_length_array[year]
            
    ##################### annual costs and unit costs ######################### 
   
    # find indexes of years when number of wells increase due to pumping rate reduction 
    # along with pumping rate and corresponding number of wells
    pumping_years = year + 1
    well_count = np.unique(num_wells)
    if min(well_count) == 0:
        well_count = np.delete(well_count, 0)
        
    added_well_count = np.zeros(len(well_count))
    for i in range(len(added_well_count)):
        if i == 0:
            added_well_count[i] = well_count[i]
        else:
            added_well_count[i] = well_count[i] - well_count[i-1]
    
    Q_vals = np.unique(Well_Q_array)
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
            if num_wells[i] - num_wells[i-1] > 0:
                Start_indx[counter] = int(i)
                counter += 1 
    
    # initialize cost arrays to track annual non-energy costs for each group of added wells 
    capital_cost_array = np.zeros((len(Start_indx), int(NUM_YEARS + WELL_LIFETIME)))
    maintenance_array = np.zeros((len(Start_indx), int(NUM_YEARS + WELL_LIFETIME)))
            
    # 1) no deepening, initial_sat_thickness < MAXIMUM_INITIAL_SAT_THICKNESS
    if initial_sat_thickness < MAXIMUM_INITIAL_SAT_THICKNESS:
        install_cost = well_unit_cost * well_length_array[0] # if no deepening, well install remains fixed 
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
                    install_cost = well_unit_cost * well_length_array[0] 
                    capital_cost_array[added_wells, year + offset] = added_well_count[added_wells] * install_cost * ((1 + INTEREST_RATE) ** WELL_LIFETIME) * INTEREST_RATE/((1 + INTEREST_RATE) ** WELL_LIFETIME-1)
                    maintenance_array[added_wells, year + offset] = MAINTENANCE_RATE * install_cost * added_well_count[added_wells] # maintenance cost [% of initial cost]
                    
                elif (year+1)/WELL_LIFETIME == 0: # Replace well every n years (well lifetime), if reduced yeild, cheaper unit cost at 200 gpm and below
                        
                    install_cost = well_unit_cost * well_length_array[year + offset]
                    capital_cost_array[added_wells, year + offset] += added_well_count[added_wells] * install_cost * ((1 + INTEREST_RATE) ** WELL_LIFETIME) * INTEREST_RATE/((1 + INTEREST_RATE) ** WELL_LIFETIME-1)
                    maintenance_array[added_wells, year + offset] += MAINTENANCE_RATE * install_cost * added_well_count[added_wells] # maintenance cost [% of initial cost]
            
                elif well_length_array[year + offset] - well_length_array[year - 1 + offset] > 0:
                    capital_cost_array[added_wells, year + offset] += added_well_count[added_wells] * install_cost * ((1 + INTEREST_RATE) ** WELL_LIFETIME) * INTEREST_RATE/((1 + INTEREST_RATE) ** WELL_LIFETIME-1)
                    capital_cost_array[added_wells, (year + offset): int((year + offset + WELL_LIFETIME))] += well_unit_cost * (well_length_array[year + offset] - well_length_array[year - 1 + offset]) * ((1 + INTEREST_RATE) ** WELL_LIFETIME) * INTEREST_RATE/((1 + INTEREST_RATE) ** WELL_LIFETIME-1) * added_well_count[added_wells]
                    install_cost = well_unit_cost * well_length_array[year + offset]
                    maintenance_array[added_wells, year + offset] += MAINTENANCE_RATE * install_cost * added_well_count[added_wells]
                    
                else:
                    capital_cost_array[added_wells, year + offset] += added_well_count[added_wells] * install_cost * ((1 + INTEREST_RATE) ** WELL_LIFETIME) * INTEREST_RATE/((1 + INTEREST_RATE) ** WELL_LIFETIME-1)
                    maintenance_array[added_wells, year + offset] += MAINTENANCE_RATE * install_cost * added_well_count[added_wells] # maintenance cost [% of initial cost]
        
                    
    ####################### annual cost metrics ###########################

    
    annual_capital_cost = np.sum(capital_cost_array, axis = 0)
    maintenance_cost = np.sum(maintenance_array, axis = 0)
    
    for year in range(pumping_years):
        well_installation_cost[year] = well_unit_cost * well_length_array[year]
        nonenergy_cost[year] = annual_capital_cost[year] + maintenance_cost[year]
        power[year] = num_wells[year] * (SPECIFIC_WEIGHT * total_head[year] * Well_Q_array[year]/EFFICIENCY)/1000 # kW 
        energy[year] = power[year] * (DAYS * 24) # kWh/year 
        energy_cost_rate[year] = ELECTRICITY_RATE # $ per kWh
        energy_cost[year] = energy[year] * energy_cost_rate[year] # $/year
        total_cost_per_well[year] = (nonenergy_cost[year] + energy_cost[year])/ num_wells[year]
        total_cost_all_wells[year] = num_wells[year] * total_cost_per_well[year] 
        
        unit_cost[year] = total_cost_all_wells[year]/volume_all_wells[year] # $/m^3
        unit_cost_per_km3[year] = unit_cost[year] * 10**9 # $/km^3
        unit_cost_per_acreft[year] = unit_cost[year] * 1233.48 # $/acft

        gridcell_year = dict(
            well_id=f'{grid_cell.Continent}{grid_cell.OBJECTID}',
            year=year,
            depth_to_piez_surface=DTW_array[year],
            aqfr_sat_thickness=sat_thickness_array[year],
            radius_of_influence=well_roi_array[year],
            drawdown=drawdown[year],
            drawdown_interference=drawdown_interference[year],  # might remove
            total_head=total_head[year],  # might remove
            well_yield=Well_Q_array[year],
            volume_per_well=volume_per_well[year],
            number_of_wells=num_wells[year],  # TODO: make file header name and variable names the same
            depleted_volume_fraction=depleted_volume_fraction[year]
        )
            
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
    for year in range(pumping_years):
    
        outputs = str(1) + ', ' + \
                  str(year+1) + ', ' + \
                  str(DEPLETION_LIMIT) + ', ' + \
                  str(grid_cell.Continent) + ', ' + \
                  str(grid_cell.CNTRY_NAME) + ', ' + \
                  str(int(grid_cell.GCAM_ID)) + ', ' + \
                  str(grid_cell.Basin_Name) + ', ' + \
                  str(grid_cell.OBJECTID) + ', ' + \
                  str(grid_cell.Area) + ', ' + \
                  str(grid_cell.Permeability) + ', ' + \
                  str(grid_cell.Porosity) + ', ' + \
                  str(grid_cell.Thickness) + ', ' + \
                  str(DTW_array[year]) + ', ' + \
                  str(initial_sat_thickness) + ', ' + \
                  str(sat_thickness_array[year]) + ', ' + \
                  str(K) + ', ' + \
                  str(T_array[year]) + ', ' + \
                  str(well_roi_array[year]) + ', ' + \
                  str(well_area_array[year]) + ', ' + \
                  str('Max_Drawdown') + ', ' + \
                  str(drawdown[year]) + ', ' + \
                  str(drawdown_interference[year]) + ', ' + \
                  str(total_head[year]) + ', ' + \
                  str(Well_Q_array[year]) + ', ' + \
                  str(volume_per_well[year]) + ', ' + \
                  str(cumulative_volume_per_well[year]) + ', ' + \
                  str(num_wells[year]) + ', ' + \
                  str(volume_all_wells[year]) + ', ' + \
                  str(cumulative_volume_all_wells[year]) + ', ' + \
                  str(available_volume) + ', ' + \
                  str(depleted_volume_fraction[year]) + ', ' + \
                  str(well_installation_cost[year]) + ', ' + \
                  str(annual_capital_cost[year]) + ', ' + \
                  str(maintenance_cost[year]) + ', ' + \
                  str(nonenergy_cost[year]) + ', ' + \
                  str(power[year]) + ', ' + \
                  str(energy[year]) + ', ' + \
                  str(energy_cost_rate[year]) + ', ' + \
                  str(energy_cost[year]) + ', ' + \
                  str(total_cost_per_well[year]) + ', ' + \
                  str(total_cost_all_wells[year]) + ', ' + \
                  str(unit_cost[year]) + ', ' + \
                  str(unit_cost_per_km3[year]) + ', ' + \
                  str(unit_cost_per_acreft[year]) + ', ' + \
                  str(grid_cell.WHYClass) + ', ' + \
                  str(grid_cell.Thickness)
                    
        if grid_cell.Index == 0 and year == 0 :
            file = open(output_path + "\\" + output_name + '.csv', 'w') 
            file.write(str(header_column_names))
            file.write('\n')
            file.write(outputs)
            file.write('\n')
            file.close()
        
        else:
            file = open(output_path + "\\" + output_name + '.csv','a')
            file.write(outputs)
            file.write('\n')
            file.close()

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

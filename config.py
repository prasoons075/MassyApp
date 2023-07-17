import json

with open('keywords.json', 'r') as f:
    keywords = json.load(f)

DEFAULT_COLUMNS = ['Sa', 'ProjectName', 'Year of Assessment (normally this year)',
       'Year Built', '# of Units', '# of floors (from filenam',
       'Project Number', 'Date', 'Description', 'Bldg',
       'Included in cells to the right?', 'Trigger count ',
       'Exclude from all? (1-Yes,0=No)', 'Chart Wording:', 'Roofing Issues',
       'Plumbing: Water Leak in Utility Closet',
       'Piping: Sanitary Not Draining', 'Plumbing Fixture Leaks',
       'Piping: Below Ground Sanitary', 'Toilet not flushing',
       'Plumbing: Sewer Smell', 'Plumbing: Water Leaks Behind Sheetrock',
       'Plumbing: Leaking Water Heaters',
       'Plumbing: Fixtures that are in poor condition', 'No hot water',
       'Low or no water pressure', 'Humidity/mold', 'Heating not working',
       'AC not working properly', 'Thermostat Issues', 'Smell in HVAC',
       'Miscellaneous HVAC', 'Fan issues', 'HVAC drip or leak',
       'Unbalanced HVAC', 'Inadequate HVAC capacity', 'Electrical', 'Fire',
       'Window issues', 'Window Leak', 'Windows Fogged', 'Window Condensation',
       'Washer/Dryer Issues', 'Dishwasher Issues']

# DEFAULT_COLUMNS = ['Description','Roofing Issues',
#        'Plumbing: Water Leak in Utility Closet',
#        'Piping: Sanitary Not Draining', 'Plumbing Fixture Leaks',
#        'Piping: Below Ground Sanitary', 'Toilet not flushing',
#        'Plumbing: Sewer Smell', 'Plumbing: Water Leaks Behind Sheetrock',
#        'Plumbing: Leaking Water Heaters',
#        'Plumbing: Fixtures that are in poor condition', 'No hot water',
#        'Low or no water pressure', 'Humidity/mold', 'Heating not working',
#        'AC not working properly', 'Thermostat Issues', 'Smell in HVAC',
#        'Miscellaneous HVAC', 'Fan issues', 'HVAC drip or leak',
#        'Unbalanced HVAC', 'Inadequate HVAC capacity', 'Electrical', 'Fire',
#        'Window issues', 'Window Leak', 'Windows Fogged', 'Window Condensation',
#        'Washer/Dryer Issues', 'Dishwasher Issues']

KEYWORDS = {
    'Roofing_Issues': keywords['Roofing_Issues'],
    'Not_Roofing_Issues': keywords['Not_Roofing_Issues'],
    'Plumbing:_Water_Leak_in_Utility_Closet': keywords['Plumbing:_Water_Leak_in_Utility_Closet'],
    'Not_Plumbing:_Water_Leak_in_Utility_Closet': keywords['Not_Plumbing:_Water_Leak_in_Utility_Closet'],
    'Piping:_Sanitary_Not_Draining': keywords['Piping:_Sanitary_Not_Draining'],
    'Not_Piping:_Sanitary_Not_Draining': keywords['Not_Piping:_Sanitary_Not_Draining'],
    'Plumbing_Fixture_Leaks': keywords['Plumbing_Fixture_Leaks'],
    'Not_Plumbing_Fixture_Leaks': keywords['Not_Plumbing_Fixture_Leaks'],
    'Piping:_Below_Ground_Sanitary': keywords['Piping:_Below_Ground_Sanitary'],
    'Not_Piping:_Below_Ground_Sanitary': keywords['Not_Piping:_Below_Ground_Sanitary'],
    'Toilet_not_flushing': keywords['Toilet_not_flushing'],
    'Not_Toilet_not_flushing': keywords['Not_Toilet_not_flushing'],
    'Plumbing:_Sewer_Smell': keywords['Plumbing:_Sewer_Smell'],
    'Not_Plumbing:_Sewer_Smell': keywords['Not_Plumbing:_Sewer_Smell'],
    'Plumbing:_Water_Leaks_Behind_Sheetrock': keywords['Plumbing:_Water_Leaks_Behind_Sheetrock'],
    'Not_Plumbing:_Water_Leaks_Behind_Sheetrock': keywords['Not_Plumbing:_Water_Leaks_Behind_Sheetrock'],
    'Plumbing_Leaking_Water_Heaters': keywords['Plumbing_Leaking_Water_Heaters'],
    'Not_Plumbing_Leaking_Water_Heaters': keywords['Not_Plumbing_Leaking_Water_Heaters'],
    'Plumbing:_Fixtures_that_are_in_poor_condition': keywords['Plumbing:_Fixtures_that_are_in_poor_condition'],
    'Not_Plumbing:_Fixtures_that_are_in_poor_condition': keywords['Not_Plumbing:_Fixtures_that_are_in_poor_condition'],
    'No_hot_water': keywords['No_hot_water'],
    'Not_No_hot_water': keywords['Not_No_hot_water'],
    'Low_or_no_water_pressure': keywords['Low_or_no_water_pressure'],
    'Not_Low_or_no_water_pressure': keywords['Not_Low_or_no_water_pressure'],
    'Humidity_mold': keywords['Humidity_mold'],
    'Not_Humidity_mold': keywords['Not_Humidity_mold'],
    'Heating_not_working': keywords['Heating_not_working'],
    'Not_Heating_not_working': keywords['Not_Heating_not_working'],
    'AC_not_working_properly': keywords['AC_not_working_properly'],
    'Not_AC_not_working_properly': keywords['Not_AC_not_working_properly'],
    'Thermostat_Issues': keywords['Thermostat_Issues'],
    'Not_Thermostat_Issues': keywords['Not_Thermostat_Issues'],
    'Smell_in_HVAC': keywords['Smell_in_HVAC'],
    'Not_Smell_in_HVAC': keywords['Not_Smell_in_HVAC'],
    'Miscellaneous_HVAC': keywords['Miscellaneous_HVAC'],
    'Not_Miscellaneous_HVAC': keywords['Not_Miscellaneous_HVAC'],
    'Fan_issues': keywords['Fan_issues'],
    'Not_Fan_issues': keywords['Not_Fan_issues'],
    'HVAC_drip_or_leak': keywords['HVAC_drip_or_leak'],
    'Not_HVAC_drip_or_leak': keywords['Not_HVAC_drip_or_leak'],
    'Unbalanced_HVAC': keywords['Unbalanced_HVAC'],
    'Not_Unbalanced_HVAC': keywords['Not_Unbalanced_HVAC'],
    'Inadequate_HVAC_capacity': keywords['Inadequate_HVAC_capacity'],
    'Not_Inadequate_HVAC_capacity': keywords['Not_Inadequate_HVAC_capacity'],
    'Electrical': keywords['Electrical'],
    'Not_Electrical': keywords['Not_Electrical'],
    'Fire': keywords['Fire'],
    'Not_Fire': keywords['Not_Fire'],
    'Window_issues': keywords['Window_issues'],
    'Not_Window_issues': keywords['Not_Window_issues'],
    'Window_Leak': keywords['Window_Leak'],
    'Not_Window_Leak': keywords['Not_Window_Leak'],
    'Windows_Fogged': keywords['Windows_Fogged'],
    'Not_Windows_Fogged': keywords['Not_Windows_Fogged'],
    'Window_Condensation': keywords['Window_Condensation'],
    'Not_Window_Condensation': keywords['Not_Window_Condensation'],
    'Washer_Dryer_Issues': keywords['Washer_Dryer_Issues'],
    'Not_Washer_Dryer_Issues': keywords['Not_Washer_Dryer_Issues'],
    'Dishwasher_Issues': keywords['Dishwasher_Issues'],
    'Not_Dishwasher_Issues': keywords['Not_Dishwasher_Issues']
}
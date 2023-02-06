filepath_prefix = '../preprocessed_data/'


separation_indexes = {'trial01': {'PPT_001' : {'left' : [2000,2500], 'right': [3000,3500], 'both' : [1000,1500]},
                                  'PPT_002' : {'left' : [1000,1500], 'right': [2000,2500], 'both' : [100,600]},
                                  'PPT_003' : {'left' : [1100,1600], 'right': [2200,2700], 'both' : [300,800]},
                                  'PPT_004' : {'left' : [1200,1700], 'right': [2200,2700], 'both' : [100,600]},
                                  'PPT_005' : {'left' : [1300,1800], 'right': [2200,2700], 'both' : [400,900]},
                                  'PPT_006' : {'left' : [1200,1700], 'right': [2200,2700], 'both' : [200,700]},
                                  'PPT_007' : {'left' : [1500,2000], 'right': [2700,3200], 'both' : [400,900]},
                                  'PPT_008' : {'left' : [1400,1900], 'right': [2500,3000], 'both' : [200,700]},
                                  'PPT_009' : {'left' : [1500,2000], 'right': [2500,3000], 'both' : [300,800]},
                                  'PPT_010' : {'left' : [1300,1800], 'right': [2200,2700], 'both' : [300,800]},
                                  'PPT_011' : {'left' : [1900,2400], 'right': [2900,3400], 'both' : [900,1400]},
                                  'PPT_012' : {'left' : [1500,2000], 'right': [2600,3100], 'both' : [300,800]},
                                  'PPT_013' : {'left' : [1600,2100], 'right': [2600,3100], 'both' : [200,700]},
                                  'PPT_014' : {'left' : [1200,1700], 'right': [2200,2700], 'both' : [100,600]},
                                  'PPT_015' : {'left' : [1500,2000], 'right': [2400,2900], 'both' : [200,700]},
                                  'PPT_016' : {'left' : [1200,1700], 'right': [2300,2800], 'both' : [100,600]},
                                  'PPT_017' : {'left' : [1300,1800], 'right': [2400,2900], 'both' : [200,700]},
                                  'PPT_018' : {'left' : [1300,1800], 'right': [2200,2700], 'both' : [300,800]},
                                  'PPT_019' : {'left' : [1500,2000], 'right': [2400,2900], 'both' : [300,800]},
                                  'PPT_020' : {'left' : [1400,1900], 'right': [2300,2800], 'both' : [200,700]}},
                      'trial02': {'PPT_001' : {'tiptoe' : [1400,1900], 'heel': [2300,2800], 'both' : [500,1000]},
                                  'PPT_002' : {'tiptoe' : [1200,1700], 'heel': [2400,2900], 'both' : [300,800]},
                                  'PPT_003' : {'tiptoe' : [1300,1800], 'heel': [2200,2700], 'both' : [300,800]},
                                  'PPT_004' : {'tiptoe' : [1200,1700], 'heel': [2200,2700], 'both' : [400,900]},
                                  'PPT_005' : {'tiptoe' : [1300,1800], 'heel': [2200,2700], 'both' : [200,700]},
                                  'PPT_006' : {'tiptoe' : [1200,1700], 'heel': [2200,2700], 'both' : [200,700]},
                                  'PPT_007' : {'tiptoe' : [1500,2000], 'heel': [2700,3200], 'both' : [400,900]},
                                  'PPT_008' : {'tiptoe' : [1300,1800], 'heel': [2500,3000], 'both' : [200,700]},
                                  'PPT_009' : {'tiptoe' : [1300,1800], 'heel': [2200,2700], 'both' : [300,800]},
                                  'PPT_010' : {'tiptoe' : [1300,1800], 'heel': [2200,2700], 'both' : [300,800]},
                                  'PPT_011' : {'tiptoe' : [1300,1800], 'heel': [2300,2800], 'both' : [200,700]},
                                  'PPT_012' : {'tiptoe' : [1300,1800], 'heel': [2200,2700], 'both' : [300,800]},
                                  'PPT_013' : {'tiptoe' : [1600,2100], 'heel': [2700,3200], 'both' : [700,1200]},
                                  'PPT_014' : {'tiptoe' : [1200,1700], 'heel': [2200,2700], 'both' : [100,600]},
                                  'PPT_015' : {'tiptoe' : [1500,2000], 'heel': [2400,2900], 'both' : [200,700]},
                                  'PPT_016' : {'tiptoe' : [1200,1700], 'heel': [2300,2800], 'both' : [100,600]},
                                  'PPT_017' : {'tiptoe' : [1300,1800], 'heel': [2300,2800], 'both' : [200,700]},
                                  'PPT_018' : {'tiptoe' : [1300,1800], 'heel': [2300,2800], 'both' : [300,800]},
                                  'PPT_019' : {'tiptoe' : [1700,2200], 'heel': [2400,2900], 'both' : [500,1000]},
                                  'PPT_020' : {'tiptoe' : [2000,2500], 'heel': [3200,3700], 'both' : [800,1300]}},
                      'trial20' : {'PPT_001' : {'left' : [1200,1700], 'right': [2200,2700], 'both' : [200,700]},
                                   'PPT_002' : {'left' : [1200,1700], 'right': [2200,2700], 'both' : [300,800]},
                                   'PPT_003' : {'left' : [1300,1800], 'right': [2300,2800], 'both' : [300,800]},
                                   'PPT_004' : {'left' : [1500,2000], 'right': [2200,2700], 'both' : [100,600]},
                                   'PPT_005' : {'left' : [1300,1800], 'right': [2200,2700], 'both' : [400,900]},
                                   'PPT_006' : {'left' : [1200,1700], 'right': [2400,2900], 'both' : [200,700]},
                                   'PPT_007' : {'left' : [1500,2000], 'right': [2200,2700], 'both' : [400,900]},
                                   'PPT_008' : {'left' : [1300,1800], 'right': [2500,3000], 'both' : [200,700]},
                                   'PPT_009' : {'left' : [1500,2000], 'right': [2400,2900], 'both' : [300,800]},
                                   'PPT_010' : {'left' : [1300,1800], 'right': [2200,2700], 'both' : [300,800]},
                                   'PPT_011' : {'left' : [1300,1800], 'right': [2200,2700], 'both' : [200,700]},
                                   'PPT_012' : {'left' : [1400,1900], 'right': [2300,2800], 'both' : [300,800]},
                                   'PPT_013' : {'left' : [1300,1800], 'right': [2100,2600], 'both' : [200,700]},
                                   'PPT_014' : {'left' : [1200,1700], 'right': [2700,3200], 'both' : [100,600]},
                                   'PPT_015' : {'left' : [1300,1800], 'right': [2200,2700], 'both' : [400,900]},
                                   'PPT_016' : {'left' : [1400,1900], 'right': [2200,2700], 'both' : [100,600]},
                                   'PPT_017' : {'left' : [1500,2000], 'right': [2300,2800], 'both' : [500,1000]},
                                   'PPT_018' : {'left' : [1300,1800], 'right': [2200,2700], 'both' : [300,800]},
                                   'PPT_019' : {'left' : [1900,2400], 'right': [2900,3400], 'both' : [800,1300]},
                                   'PPT_020' : {'left' : [1500,2000], 'right': [2500,3000], 'both' : [200,700]}}}




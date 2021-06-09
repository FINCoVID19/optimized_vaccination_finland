import numpy as np

REQUESTS = {
    "vaccination": "https://sampo.thl.fi/pivot/prod/en/vaccreg/cov19cov/fact_cov19cov.csv?row=dateweek20201226-525425&column=cov_vac_age-518413&row=area-518362&column=cov_vac_dose-533174",
    "cases_by_age": "https://sampo.thl.fi/pivot/prod/en/epirapo/covid19case/fact_epirapo_covid19case.csv?row=dateweek20200101-509030&column=ttr10yage-444309&filter=measure-444833",
    "cases_by_hcd": "https://sampo.thl.fi/pivot/prod/en/epirapo/covid19case/fact_epirapo_covid19case.csv?row=dateweek20200101-509030&column=hcdmunicipality2020-445222&filter=measure-444833",
    "cases_by_day": "https://sampo.thl.fi/pivot/prod/en/epirapo/covid19case/fact_epirapo_covid19case.csv?row=hcdmunicipality2020-445222&column=dateweek20200101-509030.509242.508804.508643.509048.509217.509192.509085.508500.508900.509052.508814.508608.508513.509190.508582.508893.509185.509166.508714.508548.508776.508992.509058.509075.509077.508981.509266.508820.508939.509213.508933.509229.509025.508506.508935.508688.509022.508974.508786.508731.509026.508689.508536.508638.509224.508882.509057.509260.508821.508656.508530.508816.509303.509040.508587.508901.509314.508984.508535.508585.509093.508999.509081.509232.508667.508613.508653.508791.508625.508551.509236.509110.508887.509195.508755.508716.508895.509134.508610.508518.509293.509312.509221.508593.509268.508858.508960.508581.508527.508562.509007.508606.509156.509250.508538.509113.508495.509121.509249.508742.508764.508493.508787.508628.508874.508878.509076.508862.508760.509237.508611.508793.508962.508711.508826.508931.509285.508788.509163.509055.509038.508888.508501.508914.509258.508682.509183.509177.508684.509315.509072.508966.508671.508744.509140.508855.508879.508851.508835.508852.508729.509291.509164.509198.508630.509005.508646.508641.509118.508672.508519.509201.509094.509010.509123.509283.509241.508512.509017.508719.508732.509263.509132.509276.509039.508778.508496.509157.508915.509259.509210.509083.509148.509044.509240.508670.509097.508906.508626.508918.508723.508775.509239.508693.509112.508836.508841.508747.509066.509203.508839.508614.508751.509146.508979.508951.508683.509152.509327.508885.508907.509111.508602.508639.509032.509326.509321.508899.508508.508525.509215.509261.508591.508664.508926.508596.508649.509084.508785.508520.509150.508929.508489.508549.508986.509011.509179.508837.509161.509202.509060.508922.508998.508706.509051.508595.509020.509046.509028.508701.509211.508650.508845.509325.508973.508674.509186.509253.509056.508889.509021.509133.508805.508603.508692.508734.509144.508569.508558.509284.508866.509172.508503.508541.509265.508663.509311.509279.508903.509300.508666.509143.508698.509054.509069.508576.509024.508896.508870.508799.508566.508627.509322.508709.508612.509071.508743.509212.508644.508956.509086.508647.508695.508758.508761.509107.509073.508579.509108.508609.508905.509141.509095.508564.508942.509178.508617.508842.508522.509047.509116.508860.508577.508753.509316.508762.508943.509270.508937.508637.508572.508812.509109.508802.508570.508497.509061.509126.509067.508946.508615.509208.509139.508597.508573.509233.508642.508545.508819.508867.508594.508678.508721.508940.509296.508868.508661.508925.508665.508838.509050.508717.508976.509182.509135.508779.508584.508794.508991.509271.508540.509216.508977.509124.508833.508537.508964.508789.509225.508620.509292.508916.509176.508631.508741.509045.508605.509245.508869.508923.509082.508567.509323.508557.508552.508952.508856.508601.509137.508930.508547.508490.509070.508988.508941.509222.508876.509209.509079.509136.508970.509220.509206.508863.508546.508533.508800.508980.509027.508910.508616.508749.509016.509255.508872.508824.508634.508763.508813.509130.509306.509153.509096.509254.508780.508773.509231.508526.508809.509138.509115.509002.508679.509264.509319.508881.508781.508619.509159.509149.509036.509304.508623.508655.508911.508774.509087.508543.508589.508554.508807.508722.508491.508676.509065.508680.508532.508892.508783.508575.508817.509287.508494.508555.509289.509301.508904.508830.508957.509169.508853.508516.509091.508607.508936.509199.509187.508801.508675.509310.508883.509189.508968.509226.508662.509298.508844.508993.508934.508912.508784.508947.509037.508685.508834.508994.508492.508880.508710.508768.509269.509100.508921.508827.508829.508708.508854.509191.508681.508769.509080.508961.508750.508746.509305.508796.509200.509053.508857.508811.508971.508542.509074.509102.509106.508806.508850.508509.509125.508958.508823.509290.509064.509278.509252.509012.508524.508660.508948.509078.509147.508514.508767.508598.509174.508759.508507.509262.509313.508995.508673.508621.508696.508897.509299.508987.508556.508972.509267.508894.508825.508840.508635.508969.508718.509006.509062.508735.509068.509101.508528.508651.509088.509272.509092.508792.508515.509003.509227.508954.508861.508690.508704.509181.508955.508694.508828.509247.508578.508950.508529.508919.509059.509214.508599.509196.509308.508997.509173.508770.508568.509218.509219.508727.509008.509043.509324.508539.508982.509033.508511.508975.508629.508571.508924.508864.508808.508636.509119.509317.509155.509131.509281.508640.509294.508748.508902.509114.508739.509098.508745.508928.509180.508847.509234.508810.508659.508865.508737.508504.509160.508848.509295.508771.508740.508920.508657.508963.508559.508677.508702.509273.509034.509019.508782.509207.509243.508658.509009.508622.508553.509014.508561.508728.509280.509041.509035.508736.509167.509158.509257.509277.508996.509063.509000.508772.508913.509274.508699.508965.508818.508531.508725.508705.508544.508766.508713.508654.508726.508583.508687.509151.508871.508849.509120.508691.508859.509184.508756.508715.508944.508832.508877.509309.509001.508632.508831.508534.508563.508565.508846.508502.508898.509318.509117.508712.509031.509307.508517.508978.509205.509175.508875.509004.509197.509104.509170.509129.508797.508669.508990.509320.508521.509251.508668.509235.509256.509042.508633.508724.509228.508815.508733.509103.509142.508803.508588.509165.508645.508891.508580.508510.508873.508700.509090.508752.509099.508590.509275.509248.508985.508908.509145.508945.508777.508757.508927.509171.508884.509023.509122.509015.509238.508983.508795.508938.509244.508765.508707.508989.509049.508600.508604.508574.509282.509288.509188.509297.508697.509246.508953.509328.509223.509029.509286.509105.509089.508720.508738.508949.509302.508703.509193.508498.508505.508843.508624.508790.508523.508917.508909.508959.508967.509168.508890.509162.509013.509018.508648.508618.508652.508560.509128.509204.508798.509194.509230.508686.508754.508499.508822.508550.509127.509154.508586.508488.&column=measure-444833#",
    'hospitalizations': "https://w3qa5ydb4l.execute-api.eu-west-1.amazonaws.com/prod/finnishCoronaHospitalData",
}

MAPPINGS = {
    "hcd_erva": {
        'Helsinki and Uusimaa Hospital District': 'HYKS',
        'Kymenlaakso Hospital District': 'HYKS',
        'South Karelia Hospital District': 'HYKS',
        'Päijät-Häme Hospital District': 'HYKS',
        'Southwest Finland Hospital District': 'TYKS',
        'Satakunta Hospital District': 'TYKS',
        'Vaasa Hospital District': 'TYKS',
        'Pirkanmaa Hospital District': 'TAYS',
        'Kanta-Häme Hospital District': 'TAYS',
        'South Ostrobothnia Hospital District': 'TAYS',
        'North Savo Hospital District': 'KYS',
        'Central Finland Hospital District': 'KYS',
        'South Savo Hospital District': 'KYS',
        'Itä-Savo Hospital District': 'KYS',
        'North Karelia Hospital District': 'KYS',
        'North Ostrobothnia Hospital District': 'OYS',
        'Central Ostrobothnia Hospital District': 'OYS',
        'Kainuu Hospital District': 'OYS',
        'Länsi-Pohja Hospital District': 'OYS',
        'Lappi Hospital District': 'OYS',
        'Åland': 'Åland',
        "All areas": "All",
        "Other areas": "Other",
    },
    "hcd_code_hcd_name": {
        0: "Åland",
        3: "Southwest Finland Hospital District",
        4: "Satakunta Hospital District",
        5: "Kanta-Häme Hospital District",
        6: "Pirkanmaa Hospital District",
        7: "Päijät-Häme Hospital District",
        8: "Kymenlaakso Hospital District",
        9: "South Karelia Hospital District",
        10: "South Savo Hospital District",
        11: "Itä-Savo Hospital District",
        12: "North Karelia Hospital District",
        13: "North Savo Hospital District",
        14: "Central Finland Hospital District",
        15: "South Ostrobothnia Hospital District",
        16: "Vaasa Hospital District",
        17: "Central Ostrobothnia Hospital District",
        18: "North Ostrobothnia Hospital District",
        19: "Kainuu Hospital District",
        20: "Länsi-Pohja Hospital District",
        21: "Lappi Hospital District",
        25: "Helsinki and Uusimaa Hospital District"
    },
    "hcd_name_hcd_code": {
        "Åland": 0,
        "Southwest Finland Hospital District": 3,
        "Satakunta Hospital District": 4,
        "Kanta-Häme Hospital District": 5,
        "Pirkanmaa Hospital District": 6,
        "Päijät-Häme Hospital District": 7,
        "Kymenlaakso Hospital District": 8,
        "South Karelia Hospital District": 9,
        "South Savo Hospital District": 10,
        "Itä-Savo Hospital District": 11,
        "North Karelia Hospital District": 12,
        "North Savo Hospital District": 13,
        "Central Finland Hospital District": 14,
        "South Ostrobothnia Hospital District": 15,
        "Vaasa Hospital District": 16,
        "Central Ostrobothnia Hospital District": 17,
        "North Ostrobothnia Hospital District": 18,
        "Kainuu Hospital District": 19,
        "Länsi-Pohja Hospital District": 20,
        "Lappi Hospital District": 21,
        "Helsinki and Uusimaa Hospital District": 25
    },
    "age_groups": {
        8: {
            "names": ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+'],
            "vaccines": {
                '0-15': '0-9',
                '16-19': '10-19',
                '20-24': '20-29',
                '25-29': '20-29',
                '30-34': '30-39',
                '35-39': '30-39',
                '40-44': '40-49',
                '45-49': '40-49',
                '50-54': '50-59',
                '55-59': '50-59',
                '60-64': '60-69',
                '65-69': '60-69',
                '70-74': '70+',
                '75-79': '70+',
                '80+': '70+',
                'All ages': 'All'
            },
            "cases": {
                '00-10': '0-9',
                '10-20': '10-19',
                '20-30': '20-29',
                '30-40': '30-39',
                '40-50': '40-49',
                '50-60': '50-59',
                '60-70': '60-69',
                '70-80': '70+',
                '80-': '70+',
                'All ages': 'All'
            },
            'population': {
                "0 - 4": "0-9",
                "5 - 9": "0-9",
                "10 - 14": "10-19",
                "15 - 19": "10-19",
                "20 - 24": "20-29",
                "25 - 29": "20-29",
                "30 - 34": "30-39",
                "35 - 39": "30-39",
                "40 - 44": "40-49",
                "45 - 49": "40-49",
                "50 - 54": "50-59",
                "55 - 59": "50-59",
                "60 - 64": "60-69",
                "65 - 69": "60-69",
                "70 - 74": "70+",
                "75 - 79": "70+",
                "80 - 84": "70+",
                "85 -": "70+",
            }
        },
        9: {
            "names": ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
            "vaccines": {
                '0-15': '0-9',
                '16-19': '10-19',
                '20-24': '20-29',
                '25-29': '20-29',
                '30-34': '30-39',
                '35-39': '30-39',
                '40-44': '40-49',
                '45-49': '40-49',
                '50-54': '50-59',
                '55-59': '50-59',
                '60-64': '60-69',
                '65-69': '60-69',
                '70-74': '70-79',
                '75-79': '70-79',
                '80+': '80+',
                'All ages': 'All'
            },
            "cases": {
                '00-10': '0-9',
                '10-20': '10-19',
                '20-30': '20-29',
                '30-40': '30-39',
                '40-50': '40-49',
                '50-60': '50-59',
                '60-70': '60-69',
                '70-80': '70-79',
                '80-': '80+',
                'All ages': 'All'
            },
            'population': {
                "0 - 4": "0-9",
                "5 - 9": "0-9",
                "10 - 14": "10-19",
                "15 - 19": "10-19",
                "20 - 24": "20-29",
                "25 - 29": "20-29",
                "30 - 34": "30-39",
                "35 - 39": "30-39",
                "40 - 44": "40-49",
                "45 - 49": "40-49",
                "50 - 54": "50-59",
                "55 - 59": "50-59",
                "60 - 64": "60-69",
                "65 - 69": "60-69",
                "70 - 74": "70-79",
                "75 - 79": "70-79",
                "80 - 84": "80+",
                "85 -": "80+",
            }
        }
    }
}

EXPERIMENTS = {
    't0': '2021-04-18',
    'vaccines_per_day': 30000,
    'simulate_T': 115,
    'r_effs': [0.75, 1.0, 1.25, 1.5],
    'init_vacc': True,
    'taus': [0.0, 0.5, 1.0],
    'num_ervas': 5,
    'num_age_groups': 9,
    'strategies': [
        ([1, 0, 0], 'Pop'),
        ([0, 0, 0], 'No vaccination'),
        ([1/3, 1/3, 1/3], 'Pop+Inc+Hosp'),
        ([1/2, 0, 1/2], 'Pop+Hosp'),
        ([1/2, 1/2, 0], 'Pop+Inc'),
        ([0, 1/2, 1/2], 'Inc+Hosp'),
        ([0, 0, 1], 'Hosp'),
        ([0, 1, 0], 'Inc'),
        (True, 'Optimized'),
    ],
    'search_num_ws': 10,
    'u_offset': 5,
}

EPIDEMIC = {
    'unreported_exponent': 2.46,
    'T_E': 1./3.,
    'T_V': 1./10.,
    'T_I': 1./4.,
    'T_q0': 1./5.,
    'T_q1': 1./3.,
    'T_hw': 1./5.,
    'T_hc': 1./9.,
    'T_hr': 1.,
    'alpha': 0.9,
    'e': 0.7,
    'delay_check_vacc': 14,
    'proportion_deaths_age': {
        8: np.array([0.0006, 0.0002, 0.0012, 0.0026, 0.0064, 0.0210, 0.0575, 0.9105]),
        9: np.array([0.0006, 0.0002, 0.0012, 0.0026, 0.0064, 0.0210, 0.0575, 0.2092, 0.7013])
    },
    'proportion_ward_age': {
        8: np.array([0.0058, 0.0107, 0.0467, 0.0605, 0.0911, 0.1450, 0.1547, 0.4855]),
        9: np.array([0.0058, 0.0107, 0.0467, 0.0605, 0.0911, 0.1450, 0.1547, 0.2008, 0.2847])
    },
    'proportion_icu_age': {
        8: np.array([0.0038, 0.0069, 0.0301, 0.0390, 0.0978, 0.2231, 0.2891, 0.3103]),
        9: np.array([0.0038, 0.0069, 0.0301, 0.0390, 0.0978, 0.2231, 0.2891, 0.2448, 0.0655])
    },
    'mu_q': {
        8: np.array([0, 0, 0, 0, 0, 0, 0, 0.1]),
        9: np.array([0, 0, 0, 0, 0, 0, 0, 0.08, 0.2])
    },
    'mu_w': {
        8: np.array([0, 0, 0, 0, 0, 0, 0, 0.3]),
        9: np.array([0, 0, 0, 0, 0, 0, 0, 0.2, 0.4])
    },
    'mu_c': {
        8: np.array([0.3, 0.1, 0.1, 0.15, 0.15, 0.22, 0.46, 0.5]),
        9: np.array([0.35, 0.1, 0.1, 0.15, 0.15, 0.22, 0.46, 0.49, 0.52])
    },
    'p_H': {
        8: np.array([0, 0, 0.02, 0.03, 0.04, 0.08, 0.16, 0.45]),
        9: np.array([0, 0, 0.02, 0.03, 0.04, 0.08, 0.16, 0.43, 0.52])
    },
    'p_c': {
        8: np.array([0, 0, 0, 0, 0, 0.01, 0.03, 0.07]),
        9: np.array([0, 0, 0, 0, 0, 0.01, 0.03, 0.05, 0.01])
    },
    'contact_matrix': {
        8: np.array(([[1.30, 0.31, 0.23, 1.07, 0.51, 0.16, 0.14, 0.09],
                      [0.28, 1.39, 0.21, 0.16, 0.87, 0.44, 0.05, 0.04],
                      [0.19, 0.19, 0.83, 0.26, 0.25, 0.42, 0.18, 0.06],
                      [0.85, 0.14, 0.25, 0.89, 0.33, 0.31, 0.24, 0.12],
                      [0.43, 0.80, 0.26, 0.36, 0.72, 0.49, 0.24, 0.17],
                      [0.12, 0.37, 0.39, 0.30, 0.44, 0.79, 0.25, 0.16],
                      [0.11, 0.04, 0.17, 0.24, 0.22, 0.25, 0.59, 0.28],
                      [0.06, 0.03, 0.05, 0.10, 0.13, 0.13, 0.23, 0.55]])),
        9: np.array(([[4.61, 1.24, 0.81, 1.71, 1.08, 0.63, 0.58, 0.15, 0.08],
                      [1.10, 7.83, 0.97, 1.02, 1.83, 0.71, 0.35, 0.13, 0.07],
                      [0.71, 0.95, 3.87, 1.84, 1.51, 1.41, 0.67, 0.19, 0.10],
                      [1.51, 1.01, 1.86, 3.25, 2.24, 1.97, 1.18, 0.21, 0.12],
                      [0.82, 1.57, 1.33, 1.94, 3.18, 2.24, 1.02, 0.29, 0.16],
                      [0.45, 0.57, 1.16, 1.60, 2.09, 2.91, 1.71, 0.26, 0.14],
                      [0.62, 0.42, 0.83, 1.43, 1.42, 2.56, 2.19, 0.72, 0.40],
                      [0.22, 0.21, 0.32, 0.36, 0.58, 0.55, 1.01, 1.09, 0.60],
                      [0.22, 0.21, 0.32, 0.36, 0.58, 0.55, 1.01, 1.09, 0.60]]))
    },
    'region_order': {
        'erva': ['HYKS', 'TYKS', 'TAYS', 'KYS', 'OYS'],
        'hcd': [
            "Southwest Finland Hospital District",
            "Satakunta Hospital District",
            "Kanta-Häme Hospital District",
            "Pirkanmaa Hospital District",
            "Päijät-Häme Hospital District",
            "Kymenlaakso Hospital District",
            "South Karelia Hospital District",
            "South Savo Hospital District",
            "Itä-Savo Hospital District",
            "North Karelia Hospital District",
            "North Savo Hospital District",
            "Central Finland Hospital District",
            "South Ostrobothnia Hospital District",
            "Vaasa Hospital District",
            "Central Ostrobothnia Hospital District",
            "North Ostrobothnia Hospital District",
            "Kainuu Hospital District",
            "Länsi-Pohja Hospital District",
            "Lappi Hospital District",
            "Helsinki and Uusimaa Hospital District"
        ]
    }
}

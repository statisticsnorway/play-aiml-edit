# Beskrivelse av kode


### config.py
- Python klassen Config har oversikt over ulike parametere (bøtter, kolonner i datasett, hvilke år vi skal se på, type akkumuleringsfeil, splitting av trening/validering)

### load_data.py
- load_data_year henter ut VHI data for et gitt år og lagrer det som en Pandas dataframe. 
- get_all_data brukes for å hente ut VHI data, hvilke år man ønsker å hente ut styres av parameteren "year" i Config. 

### create_accumulation_error.py
- Python klassen AccumulationErrors blir brukt for å generere akkumuleringsfeil i datasettet. Foreløpig blir det laget 3 ulike akkumuleringsfeil (se kode for beskrivelse):
    - Kumulativ sum (cumulative_sum)
    - Kaskadefeil (cascading_error)
    - Tilfeldig feil (random)


### base_model.py
- Generisk klasse for ulike maskinlæringsmodeller, brukt for initialisering av parametere og regne ut hvor bra modellen gjør det (f1, f2, f_beta, precision, recall). 


### iso_forest.py


### lgb_model.py

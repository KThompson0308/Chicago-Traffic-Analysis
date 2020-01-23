# Analysis of Traffic Crashes in Chicago Since 2013

[URL for Chicago Crash Data](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if)


## Attributes and Definitions

* __RD_NO__ - Chicago Police Department report number. This number can be used to link to the same crash in the Vehicles and People datasets. This number also serves as a unique ID in this dataset.

* __CRASH_DATE_EST_I__ - Crash date estimated by desk officer or reporting party (only used in cases where crash is reproted at police station days after the crash)

* __CRASH_DATE__ - Date and time of crash as entered by the reporting officer.

* __POSTED_SPEED_LIMIT__ - Posted speed limit, as determined by reporting officer. (strange speed limits)

* __TRAFFIC_CONTROL_DEVICE__ - Traffic control device present at crash location, as determined by reporting officer. (unclear)

* __DEVICE_CONDITION__ - Condition of traffic control device, as determined by reporting officer.

* __WEATHER_CONDITION__ - Weather condition at time of crash, as determined by reporting officer.

* __LIGHTING_CONDITION__ - Light condition at time of crash, as determined by reporting officer.

* __FIRST_CRASH_TYPE__ - Type of first collision in crash

* __TRAFFICWAY_TYPE__ - Trafficway type, as determined by reporting officer.

* __LANE_CNT__ - Total number of through lanes in either direction, excluding turn lanes, as determined by reporting officer (0=intersection)

* __ALIGNMENT__ - Street alignment at crash location, as determined by reporting officer.

* __ROADWAY_SURFACE_COND__ - Road surface condition, as determined by reporting officer.

* __REPORT_TYPE__ - Administrative report type (at scene, at desk, amended)

* __CRASH_TYPE__ - A general severity classification for the crash (Either Injury and/or Tow Due to Crash or No Injury / Drive Away)

* __INTERSECTION_RELATED_I__ - A field observation by the police officer whether an intersection played a role in the crash. Does not represent whether or not the crash occurred within the intersection.

* __NOT_RIGHT_OF_WAY_I__ - Whether the crash begun or first contact was made outside of the public right-of-way.

* __HIT_AND_RUN_I__ - Crash did/did not involve a driver who caused the crash and fled the scene without exchanging information and/or rendering aid.

* __DAMAGE__ - A field observation of estimated damage.

* __DATE_POLICE_NOTIFIED__ - Calendar date on which police were notified of the crash.

* __PRIM_CONTRIBUTORY_CAUSE__ - The factor which was most significant in causing the crash, as determined by officer judgment.

* __SEC_CONTRIBUTORY_CAUSE__ - The factor which was second most significant in causing the crash, as determined by officer judgment.

* __STREET_NO__ - Street address number of crash location, as determined by reporting officer.

* __STREET_DIRECTION__ - Street address direction (N,E,S,W) of crash location, as determined by reporting officer.

* __STREET_NAME__ - Street address name of crash location, as determined by reporting officer.

* __BEAT_OF_OCCURRENCE__ - Chicago police department beat id. [Boundaries URL](https://data.cityofchicago.org/d/aerh-rz74)

* __PHOTOS_TAKEN_I__ - Whether the Chicago Police Department took photos at the location of the crash.

* __STATEMENTS_TAKEN_I__ - Whether statements were taken from unit(s) involved in crash.

* __DOORING_I__ - Whether crash involved a motor vehicle occupant opening a door into the travel path of a bicyclist, causing a crash.

* __WORK_ZONE_I__ - Whether the crash occurred in an active work zone.

* __NUM_UNITS__ - Number of units involved in the crash. A unit can be a motor vehicle, a pedestrian, a bicyclist, or another non-passenger roadway user. Each unit represents a mode of traffic with an independent trajectory.

* __MOST_SEVERE_INJURY__ - Most severe injury sustained by any person involved in the crash.

* __INJURIES_TOTAL__ - Total persons sustaining fatal, incapacitating, non-incapacitating, and possible injuries as determined by the reporting officer.

* __INJURIES_FATAL__ - Total persons sustaining fatal injuries in the crash.

* __INJURIES_INCAPACITATING__ - Total persons sustaining incapacitating/serious injuries in the crash, as determined by reporting officer. An incapacitating injury is any non-fatal injury that prevents the injured person from walking, driving, or normally continuing the activities they were capable of performing before the injury occurred. Includes severe lacerations, broken limbs, skull or chest injuries, and abdominal injuries.

* __INJURIES_NON_INCAPACITATING__ - Total persons sustaining non-incapacitating injuries in the crash as determined by reporting officer. Any non-fatal, non-incapacitating injury that is evident to observers at the scene of the crash. Includes lump on head, abrasions, bruises, and minor lacerations.

* __INJURIES_REPORTED_NOT_EVIDENT__ - Total persons sustaining posssible injuries in the crash as determined by the reporting officer. Includes momentary unconsciousness, claims of injuries not evident, limping, complaint of pain, nausea, and hysteria.

* __INJURIES_NO_INDICATION__ - Total persons sustaining no injuriesi n the crash as determined by reporting officer.

* __CRASH_DAY_OF_WEEK__ - The day of the week component of CRASH_DATE. Sunday=1

* __CRASH_MONTH__ - The month component of CRASH_DATE

* __LATITUDE__ - The latitude of the crash location, as determined by the reporting officer.

* __LONGITUDE__ - The longitude of the crash location, as determined by reporting officer.

* __LOCATION__ - The crash location, as determined by the latitude and longitude. 


UP-TO-DATE TYPICAL SHIFT OF POLICE OFFICERS CAN BE FOUND AT https://chicago.suntimes.com/city-hall/2019/10/7/20903404/chicago-police-department-officers-start-times-rescinded-fraternal-order-union

GOOGLE ROADS API for retrieving speed limits for dataset. https://developers.google.com/maps/documentation/roads/speed-limits
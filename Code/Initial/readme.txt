The Agent Based Model for Easter Island (code in python, file name ‘easter_island_abm.py’) 
contains 2 agents: trees and households. Each household consist of a variable number of people. 
When a household reaches a population of 100, it divides into 2 households of 50 persons each,
 the newly created household changes its position on the island in reference to the "parent household". 
The number of trees is assumed to be the essential and most important resource on the Island. Each 
individual household requires a tree per year per individual for surviving, if this amount cannot 
be harvested the household changes its location. If a household can not harvest the required amount 
of trees for multiple years in a row it vanishes. Households can only harvests threes within a certain 
distance (1 km). The population grows at a rate of 6%. There also exists a reproduction rate for the 
trees but it is currently not used (thus in the current version trees do not growth, they are just 
there and they are consumed by the human population).

The initial conditions for the attached results: initial population on the island is of 60 people divided 
into 3 different households (3 household of 20 peaple each), the initial number of trees is 1 million. As 
mentioned, tree harvesting range is a circle of 1 km radius, and the magnitude of movement of a household 
is 5 km.

The code needs to be optimised because the current version of the model is very time-intensive (the results 
in the attached simulation have required at least 6 hours to complete on a laptop). It should be possible 
to speed-up the simulation significantly by somehow reducing the amount of trees to go through at each time 
step during the update loop. Another improvement could be to plot only a fraction of the results.

Another improvement could be to change the shape of the island from the current unit-square to a triangle 
which is more similar to the real shape of the island.


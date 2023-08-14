# A Spatially Explicit Agent-Based Model of Human-Resource Interaction on Easter Island
## Masterthesis by Peter Steiglechner

as part of the [COSSE Master](https://www.kth.se/en/studies/master/computer-simulations-for-science-and-engineering/msc-computer-simulations-for-science-and-engineering-1.44243) at KTH Stockholm and TU Berlin, 2018 - 2020.

***
Associated publication: 

Steiglechner, P., Merico, A. (2022). Spatio-Temporal Patterns of Deforestation, Settlement, and Land Use on Easter Island Prior to European Arrivals. In: Rull, V., Stevenson, C. (eds) The Prehistory of Rapa Nui (Easter Island). Developments in Paleoenvironmental Research, vol 22. Springer, Cham. https://doi.org/10.1007/978-3-030-91127-0_16
***

<img src="COSSE_workshop_tafel" alt="drawing" width="600"/>
Sketch created by a participant during a workshop of the COSSE master programme, where I presented the model.

### Overview:

The folder `Code` contains all necessary scripts to reproduce the simulations used for the thesis.

In the `Full_Model` subfolder running 
```
python FullModel.py seed lowFix noRegrowth linear NormPop alphaStd
```
where `seed` is replaced by a number, `lowFix` means a low nitrogen fixation scenario, `noRegrowth` means a scenario in which trees do not regrow, `linear` means that the tree preference of household agents changes linearly with the tree density, `NormPop` is the default for the population growth function, and `alphaStd` means that during moving an agent considers all penalties (lake distance, lack of tree availability, elevation and slope, lack of free arable land, and population density) are weighted equally.

The folder `Thesis` contains all .tex documents to create the thesis. A compiled PDF version can be found in `Thesis/Thesis_final_EasterIslandABM_Masterarbeit_30Jun2020_PeterSteiglechner.pdf`. The folder Presentation contains the presentation for the Master Thesis defence.

### Abstract
The history of Easter Island, with its cultural and ecological mysteries, has attracted the interests of archaeologists, anthropologists, ecologists, and economists alike. Despite the great scientific efforts, uncertainties in the available archaeological and palynological data leave a number of critical issues unsolved and open to debate. The maximum size reached by the human population before the arrival of Europeans and the temporal dynamics of deforestation are some of the aspects still fraught with controversies. By providing a quantitative workbench for testing hypotheses and scenarios, mathematical models are a valuable complement to the observational-based approaches generally used to reconstruct the history of the island. Previous modelling studies, however, have shown a number of shortcomings in the case of Easter Island, especially when they take no account of the stochastic nature of population growth in a temporally and spatially varying environment. Here, I present a new stochastic, Agent-Based Model characterised by (1) realistic physical geography of the island and other environmental constraints (2) individual agent decision-making processes, (3) non-ergodicity of agent behaviour and environment, and (4) randomised agent-environment interactions. I use the model and the best available data to determine plausible spatial and temporal patterns of deforestation and other socioecological features of Easter Island prior to the European contact. I further identify some non-trivial connections between microscopic decisions or constraints (like local confinement of agents' actions or their adaptation strategy to environmental degradation) and macroscopic behaviour of the system that can not easily be neglected in a discussion about the history of Easter Island before European contact.

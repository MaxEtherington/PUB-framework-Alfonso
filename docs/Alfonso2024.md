Spatio-temporal copper prospectivity in the American Cordillera
predicted by positive-unlabeled machine learning
Christopher P. Alfonso1,†, R. Dietmar Müller1, Ben Mather1, and Michele Anthony2
1
 EarthByte Group, School of Geosciences, Building F09, University of Sydney, Sydney, New South Wales 2006, Australia
2
 BHP Metals Exploration, 6840 N Oracle Rd, Suite 100, Tucson, Arizona 85704, USA
ABSTRACT
Porphyry copper deposits contain the
majority of the world’s discovered mine-
able reserves of copper. While these deposits
are known to form in magmatic arcs along
subduction zones, the precise contributions
of different factors in the subducting and
overriding plates to this process are not well
constrained, making predictive prospectivity
mapping difficult. Empirical machine learn-
ing-based approaches to this problem have
been explored in the past but are hampered
by the lack of comprehensive labeled data for
training classification models.
Here we present a model trained using
a semi-supervised positive-unlabeled (PU)
learning algorithm, trained using only one
set of labeled data: known deposit locations.
Time-dependent and present-day mineral
prospectivity maps created using the classi-
fier show the past evolution and present-day
state of porphyry copper mineralization in
the American Cordillera, with several zones
of high predicted prospectivity unrelated to
any known deposits presenting potential op-
portunity for future exploration targeting.
Feature importance and partial depen-
dence analysis shed light on the complex
mechanisms behind porphyry copper forma-
tion, identifying thick arc crust, rapid con-
vergence, and a sufficient supply of volatile
fluids into the subduction system as the pri-
mary prerequisites for mineralization. Sig-
nificantly different results between models
trained on data from North or South Amer-
ica suggest the existence of extensive variety
among porphyry copper provinces.
High values of performance metrics for
North America, including receiver operating
characteristic area-under-the-curve (ROC
Christopher P. Alfonso https://orcid.org/0009
-0009-6438-2945
†christopher .alfonso@sydney .edu .au
AUC), indicate that PU models are capable of
exhibiting equal or better performance when
compared to traditional classifiers. How-
ever, relatively poor metric scores for South
American data demonstrate that model per-
formance is not necessarily uniform across
different tectonic settings and care should,
therefore, be taken when applying the PU
method to new areas. Nonetheless, the meth-
ods developed here are expected to be appli-
cable to data-poor regions and time periods
across the globe, potentially identifying many
more potential targets for porphyry copper
exploration.
INTRODUCTION
Demand for certain economically crucial
mineral resources, including so-called “critical
minerals” and “high-tech metals,” is expected to
increase dramatically in the near future (Depart-
ment of Regional NSW, 2021). The resource
with the greatest projected demand is copper,
and much of the world’s known copper reserves
are found in porphyry copper deposits, so depos-
its of this type are a key area of interest for future
exploration. However, despite decades of exten-
sive research into the subject, the exact nature
of the processes behind the formation of these
deposits is imperfectly understood (e.g., Lee
and Tang, 2020; Park et al., 2021; Sillitoe, 2010;
Wilkinson, 2013), and increasingly sophisticated
prospectivity models will be needed to allow the
identification of as-yet undiscovered porphyry
copper systems.
Porphyry Copper Systems
Porphyry copper systems are one of the three
most significant types of copper deposits glob-
ally, along with the sediment-hosted stratiform
and iron oxide-copper-gold types. Of these three
primary types, porphyry deposits are the most
common, making up ∼70% of known copper
reserves worldwide (Sillitoe, 2012), and often
hosting additional metals such as gold, molyb-
denum, and silver (Loucks, 2014; Sillitoe, 2010).
Deposits of this type are formed by subduction-
related processes, and as such are commonly
found in linear belts located at current or former
convergent plate boundaries (Sillitoe, 2010). For
example, the porphyry deposits of the Macquarie
Arc of eastern Australia hold upward of 80 Moz
of gold and 13 Mt of copper (Glen et al., 2012;
Kreuzer et al., 2015), while the Pacific Ocean
is ringed by a series of porphyry belts hosting
a combined copper endowment of ∼1800 Mt
(Sillitoe, 2012). Despite their association with
convergent tectonic environments, structural
analysis suggests that porphyry copper depos-
its are commonly formed in low-strain envi-
ronments, which would theoretically constrain
them to locations or times at which significant
deformation is absent, such as during periods of
stress relaxation in the magmatic arc (Tosdal and
Richards, 2001).
Porphyry copper systems are formed by
hydrothermal alteration of intermediate-felsic
porphyry intrusions such as stocks and dikes,
associated with and fed by an underlying com-
posite pluton (Sillitoe, 2010). Generally, the
magmas that make up these intrusions have
been found to be relatively water-rich due to long
periods of fractionation and replenishment in a
high-pressure magma chamber, changing the
order of plagioclase, hornblende, and magnetite
crystallization and producing a distinctive trace
element signature: enrichment of strontium and
vanadium and depletion of scandium and yttrium
(Loucks, 2014). In theory, this geochemical sig-
nature could be used to distinguish between
ordinary and potential porphyry copper-forming
arc magmas and thus prove useful in exploration.
However, it has also been proposed that fertile
arc magmas capable of producing porphyry
copper deposits might in fact be fairly common.
According to this model, the rarity of mineable
deposits of this type would instead be explained
by the low probability of a deposit reaching a
reasonable size and then being exhumed to a
GSA Bulletin; January/February 2025; v. 137; no. 1/2; p. 702–711; https://doi.org/10.1130/B37614.1.
Published online 29 August 2024
702
For permission to copy, contact editing@geosociety.org
© 2024 Geological Society of America
Downloaded from http://pubs.geoscienceworld.org/gsa/gsabulletin/article-pdf/137/1-2/702/7082726/b37614.1.pdf
by Australian National University user
Copper prospectivity in the American Cordillera
shallow enough depth to enable its discovery
(Richards, 2022), the deposit being eroded to
the point of destruction. In this case, it may be
impossible to identify deposits suitable for min-
ing using these geochemical markers alone.
Plate Tectonics of the North American
Cordillera
The North American Cordillera, the vast oro-
gen that makes up much of the western portion
of the continent, constitutes one of the world’s
largest porphyry copper provinces, second only
to the Andes of South America (Sillitoe, 2012).
Many global and regional plate tectonic recon-
structions (e.g., Engebretson et al., 1985; Müller
et al., 2019; Seton et al., 2012) depict a relatively
simple subduction system along this continental
margin. According to these models, an enormous
Farallon plate, making up approximately one
third of the Pacific Ocean, subducted under the
continental margin at a continuous plate bound-
ary stretching from the Andes to Alaska in a con-
figuration that remained essentially unchanged
since ca. 200 Ma.
These relatively simple models are contra-
dicted by the extensive geologic evidence for a
much more complex subduction system in west-
ern North America. The North American Cordil-
lera is made up of numerous terranes of differing
origins, including several oceanic arc complexes
indicating the presence of intra-oceanic subduc-
tion (Dickinson, 2004, 2008; Ingersoll, 2008;
Ingersoll and Schweickert, 1986; Moores, 1998).
Furthermore, recent developments in seismic
tomography have allowed the detailed imaging
of the mantle structure beneath North America,
revealing the presence of at least two large struc-
tures in the region interpreted as subducted slabs
(Sigloch, 2011; Sigloch et al., 2008). If correct,
this interpretation indicates the presence of a
vanished network of Jurassic–Cretaceous sub-
duction zones several thousand kilometers west
of the contemporary North American continent.
This discovery has led to several new tectonic
reconstructions incorporating these tomographi-
cally imaged plate boundaries (e.g., Sigloch and
Mihalynuk, 2013, 2017), representing many of
the Cordilleran terranes as exotic island arcs and
continental fragments.
Most recently, Clennett et al. (2020) pro-
posed a new continuously closing plate model
for western North America since 170 Ma; this
model includes subduction zones informed by
the aforementioned geologic and tomographic
data, with closed tectonic plates formed by
constructing additional divergent and transform
plate boundaries according to the general rules
of plate tectonics. The model, which forms the
basis of this work, is in good agreement with
the available seismic tomography and paleo-
magnetic data. However, debate persists over the
existence of these plate boundaries, the orienta-
tion of subduction, and the nature of the Cordil-
leran terranes (e.g., Gehrels et al., 2017; Lowey,
2017, 2023; Tikoff et al., 2022; Yokelson et al.,
2015). In particular, the exotic origins assigned
to the terranes by this and other similar mod-
els appear to be incompatible with other lines
of evidence indicating a North American origin
for several terranes, which have led to alterna-
tive interpretations involving the opening and
subsequent closing of back-arc basins on the
North American margin (e.g., Western Klamath
terrane; LaMaskin et al., 2022).
Mapping Porphyry Copper Prospectivity
Using Machine Learning
In recent years, machine learning has emerged
as a key area of interest for prospectivity map-
ping research. Machine learning techniques
have been applied at a diverse range of scales,
from local (Carranza and Laborte, 2015; Zuo
and Carranza, 2011) to continental (Butterworth
et al., 2016; Diaz-Rodriguez et al., 2021). Pro-
spectivity mapping is generally approached
as a binary classification problem; according
to this framework, a given point in space, and
potentially time, is designated as a positive or
negative observation according to the presence
or absence, respectively, of the type of mineral
deposit being investigated. However, in order
to perform a classification, traditional machine
learning methods such as random forest (e.g.,
Butterworth et al., 2016; Carranza and Laborte,
2015; Diaz-Rodriguez et al., 2021), support
vector machine (Butterworth et al., 2016; Diaz-
Rodriguez et al., 2021; Zuo and Carranza, 2011),
and neural network (Diaz-Rodriguez et al., 2021)
classifiers require both positive and negative data
points to train a model.
While positive observations can be obtained
relatively easily from maps or databases of
known mineral occurrences, there is no widely
accepted method for the identification of suitable
negative, non-prospective data points. For exam-
ple, one suggested method is to simply choose
random locations greater than a certain distance
from known deposits, assuming that these loca-
tions are most likely non-prospective (Carranza
et al., 2008). Other methods include obtaining
negative training data from the locations of min-
eral deposits of a different type to that which is
being investigated (Nykänen et al., 2015) or drill
core sites where no identifiable mineralization
has been reported (Lindsay et al., 2022). The
viability of these methods, however, is dimin-
ished at larger scales and coarser spatial resolu-
tions, while drill sites cannot often be assigned
a meaningful age, rendering them incompatible
with any spatio-temporal approach.
At a continental to global scale, a recent study
of the porphyry deposits of South America
(Butterworth et al., 2016) used the locations of
known mineral deposits and assigned random
ages—different than their true formation ages—
to create non-deposit data points, reasoning that
these represented locations where deposit forma-
tion was theoretically viable but at a time when
it did not actually occur. Diaz-Rodriguez et al.
(2021), meanwhile, applied the set of geochemi-
cal magma prospectivity metrics developed by
Loucks (2014) to entries in the Geochemistry of
Rocks of the Oceans and Continents (GEOROC)
database (https://georoc .eu/) for North and
South America, in order to identify non-pro-
spective intrusive rocks within the study area.
The authors of both studies identified multiple
potential issues with their approaches. However,
the task of selecting negative data points can be
avoided entirely by using a positive-unlabeled
(PU) classifier (e.g., Bekker and Davis, 2020),
which requires labels only for a small set of
positive points, while the majority of the train-
ing data, including all of the negative class, may
be left unlabeled.
Traditionally, the features used to estimate
porphyry copper prospectivity have related
solely to the overriding plate in the subduction
system (e.g., Loucks, 2014). While some more
recent works have examined factors relating to
the subducting plate (Butterworth et al., 2016;
Diaz-Rodriguez et al., 2021), these analyses
have focused on this plate to the exclusion of the
overriding plate. For this work, however, data-
sets from both the subducting and overriding
plate were considered in the analysis.
METHODS
Plate Model
The plate tectonic reconstruction used
throughout this work was developed from that of
Clennett et al. (2020). This model was an exten-
sion of the global model of Müller et al. (2019),
with a focus on producing a more detailed tec-
tonic reconstruction of the North American Cor-
dillera. Further minor modifications were made
to the model for this work, using the GPlates
software package (https://www .gplates .org/)
and its Python application programming inter-
face, pyGPlates.
Firstly, several subduction zones in the pub-
lished plate model have no associated motions,
instead remaining entirely stationary relative to
the mantle. This was judged to be infeasible, as
geodynamic research indicates that trenches and
subducting slabs are almost always in motion
Geological Society of America Bulletin, v. 137, no. 1/2 703
Downloaded from http://pubs.geoscienceworld.org/gsa/gsabulletin/article-pdf/137/1-2/702/7082726/b37614.1.pdf
by Australian National University user
Alfonso et al.
relative to the mantle, exhibiting retreat or—
more rarely—advance (e.g., Capitanio, 2013;
Capitanio et al., 2010; Schellart et al., 2007;
Stegman et al., 2010). As trenches of interme-
diate length generally display relatively slow
retreat (Schellart, 2008), a small amount of
trench rollback was imposed on the subduc-
tion zones in this model, while ensuring that
they remained located above their associated
tomographically imaged slabs. Additionally,
the Orcas plate (170–130 Ma), located within
the Cordilleran archipelago, was split into two
sections separated by an east-west trending
mid-ocean ridge, consistent with the subsequent
plate configuration after 130 Ma. This was done
to improve the consistency of the plate motion
model, which appeared to indicate divergence
within the Orcas plate and convergence at its
boundaries. Finally, the reconstruction’s abso-
lute plate motion model was constrained using
the iterative optimization workflow developed
by Tetley et al. (2019).
Training Data
Several different datasets, relating to both
the subducting and overriding tectonic plates,
were used for the analysis. Subduction-related
features were extracted according to the method
of Diaz-Rodriguez et al. (2021). These features
included kinematic parameters, such as plate
convergence velocity and subduction obliquity,
in addition to estimates of seafloor age after Wil-
liams et al. (2021), silicic and carbonate sedi-
ment thickness (Dutkiewicz et al., 2019; Dutkie-
wicz et al., 2017), sediment pore water volume
density (Athy, 1930), and carbon dioxide (CO2)
content in the upper oceanic crust (Müller and
Dutkiewicz, 2018). Crustal thickness in the
overriding plate was approximated by topogra-
phy, extracted from the paleogeographic recon-
struction of Cao et al. (2017), translated into the
appropriate plate reconstruction. The magnetic
anomaly in the vicinity of each data point was
obtained from the EMAG2v3 dataset (Meyer
et al., 2017), a 2 arc-minute resolution global
grid published by the U.S. National Oceanic and
Atmospheric Administration.
In addition to the instantaneous time-depen-
dent variables outlined above, a number of
cumulative quantities were calculated. For each
of the silicic and carbonate sediment thickness
and water density datasets, the values at subduc-
tion zone locations were extracted, multiplied by
the local convergence rate, and integrated over
time. In this manner, estimates for the total vol-
ume of sediment and water subducted into the
upper mantle since the beginning of the plate
model were obtained, functioning as a proxy for
mantle enrichment.
427 positive observations, representing
known porphyry copper deposit formation loca-
tions and times, were obtained from the dataset
used by Diaz-Rodriguez et al. (2021); negative
data points were discarded when training PU
models. Unlabeled observations, which could
conceivably be either positive or negative, were
generated randomly for each time step using
a uniform distribution across the surface of
Earth, then restricted to locations less than 6°
(∼667 km) inboard of an active North or South
American subduction zone. The final training
dataset contained 100 unlabeled observations
for each 1 m.y. timestep, for a total of 17,000
across the entire model time frame. Additional
subsets of the data were extracted by select-
ing only those observations above or below the
latitude of 12°N, representing North and South
America, respectively.
Classification Model and Prospectivity
Mapping
Time-dependent mineral prospectivity maps
were generated by a PU classifier using the boot-
strap aggregating “bagging” method developed
A B
C D
Figure 1. Snapshots of the time-dependent porphyry copper prospectivity maps, taken at
160 Ma, 140 Ma, 120 Ma, and 100 Ma. Mid-ocean ridges and other plate boundaries are
shown in red and black, respectively, while known deposit locations used to train the model
are indicated by white stars. Orthographic projection, centered at 10°N, 100°W. ALI—Alis-
itos; ALK—Alaska; ANG—Angayucham; CAR—Caribbean; CHA—Chazca; FAR—Far-
allon; GRO—Guerrero; INS—Insular; IZA—Izanagi; MEZ—Mezcalera; NAM—North
America; ORC—Orcas; PAC—Pacific; PHX—Phoenix; SAM—South America.
704 Geological Society of America Bulletin, v. 137, no. 1/2
Downloaded from http://pubs.geoscienceworld.org/gsa/gsabulletin/article-pdf/137/1-2/702/7082726/b37614.1.pdf
by Australian National University user
Copper prospectivity in the American Cordillera
by Mordelet and Vert (2014), in which a PU clas-
sifier can be constructed from an ensemble of
regular binary classifiers. Several different base
classification algorithms were tested, including
random forest, adaptive boosting (AdaBoost),
and gradient boosting. The random forest and
AdaBoost algorithms are included within the
Scikit-learn Python package (Pedregosa et al.,
2011), while gradient boosting was implemented
using XGBoost (Chen and Guestrin, 2016). The
pulearn (https://pulearn .github .io /pulearn) pack-
age was used to create the PU bagging classifier.
The probability, ranging from 0% to 100%, of
a given observation belonging to the positive
“deposit” class was determined using the clas-
sifier and used to quantify porphyry copper pro-
spectivity. This metric was applied to a global
0.5°-resolution grid of points generated at 1 m.y.
timesteps to create time-dependent prospectiv-
ity maps, showing the likelihood of a porphyry
copper deposit forming at a given time and loca-
tion (Figs. 1 and 2). Finally, to isolate the effect
of the PU learning method on the results, this
workflow was repeated using a support vector
machine (SVM) trained directly on the dataset
of Diaz-Rodriguez et al. (2021).
RESULTS AND DISCUSSION
Selected snapshots of the time-dependent
deposit probability maps are shown in Figures 1
and 2, demonstrating the evolution of the mod-
eled Cordilleran porphyry copper prospectivity.
These maps were produced using the PU model
with a random forest as the base classifier, though
all of the algorithms tested produced broadly
similar results (Table S11). High prospectivity
values were largely constrained to the continen-
tal margins of North and South America, while
the model predicted consistently low prospectiv-
ity for the North American island arcs.
While many of the high-prospectivity loca-
tions and times identified by the model appear
to be associated with known porphyry deposits
(e.g., SW USA–NW Mexico, 50 Ma; Fig. 2B),
several of these zones do not contain any iden-
tified deposits (e.g., NW Mexico at 75 Ma and
25 Ma; Figs. 2A and 2C, respectively). If arc-
related rocks of the corresponding ages could be
found within these areas, they would likely make
promising candidates for further, more targeted
mineral exploration techniques.
A B
C D
Figure 2. As in Figure 1, snapshots of the prospectivity maps, taken at 75 Ma, 50 Ma, 25 Ma,
and 0 Ma. CAR—Caribbean; COC—Cocos; FAR—Farallon; JDF—Juan de Fuca; KRO—
Kronotsky; KUL—Kula; NAM—North America; NAZ—Nazca; ORC—Orcas; PAC—Pacific;
SAM—South America; VAN—Vancouver.
1Supplemental Material. Table S1, Figures S1–
S3, and Animation S1. Data and Python code to
reproduce this work can be accessed from Zenodo
(https://doi .org /10 .5281 /zenodo .8157690) and GitHub
(https://github .com /cpalfonso /stellar-data-mining),
respectively. Please visit https://doi .org /10 .1130 /GSAB
.S.26312101 to access the supplemental material;
contact editing@geosociety .org with any questions.
Feature Importance
To assess the impact of individual features on
the classifier, feature importance was calculated
for each input parameter (Fig. 3). Boxplots illus-
trating the distributions of values for the six most
important features are shown in Figure 4, while
partial dependence plots (Fig. 5) were also con-
structed for these features in order to isolate and
visualize the mean effect of each on the predicted
probability. The mean overriding plate crustal
thickness (sampled from a 2-arc-degree radius
from a given point) showed the largest impor-
tance value by a significant margin. Thicker crust
was strongly associated with porphyry deposits:
the average mean crustal thickness value for all
positive observations was 46.6 km, compared
with 38.6 km for the random unlabeled points
(Fig. 4A), while Figure 6A indicates a dramatic
increase in predicted probability with increasing
crustal thickness. This result is consistent with
recent research indicating the presence of an
arc thick enough to allow garnet fractionation
(>45 km) to be the main prerequisite for the
formation of porphyry copper deposits, due to
the resulting increase in the oxidation level and
thus copper transport capability of the magmatic
fluids (Lee and Tang, 2020).
The distance to the subduction zone was iden-
tified as another important feature. Due to the
method used to randomly generate the unlabeled
observations, these points exhibited a uniform
Geological Society of America Bulletin, v. 137, no. 1/2 705
Downloaded from http://pubs.geoscienceworld.org/gsa/gsabulletin/article-pdf/137/1-2/702/7082726/b37614.1.pdf
by Australian National University user
Alfonso et al.
Figure 3. Model feature impor-
tance (mean decrease in impu-
rity) for the six highest-ranked
features identified in Figure 3,
normalized relative to the most
important feature, calculated
for the models trained on the
full dataset for both regions
(All), North American (NA)
data only, and South American
(SA) data only.
distribution in this variable, while deposits were
generally found at distances greater than 200 km
(mean values: positive 416 km, random 277 km;
Figs. 4B and 5B). As porphyry copper deposits
are found in magmatic arcs, this threshold most
likely represents a minimum distance from the
plate boundary for the formation of an arc in
general, rather than any process specific to por-
phyry deposits.
A high feature importance value was also
indicated for the orthogonal component of the
subduction convergence rate, with porphyry
deposits found to be associated with a high
value for this parameter (mean values: positive
9.02 cm yr−1, random 5.62 cm yr−1; Figs. 4C and
5C). This is in good agreement with the findings
of Diaz-Rodriguez et al. (2021), who used the
same training dataset but a different plate motion
model and learning algorithm and also observed
that porphyry copper deposits were associated
with high convergence rates. Additionally, a
potential relationship between these deposits
and strongly compressional tectonic environ-
ments featuring significant crustal deformation
and uplift has previously been suggested by Sil-
litoe (2010) and Loucks (2014).
Trench velocity, the fourth identified param-
eter, indicates trench retreat or advance relative
to the subducting plate, represented by nega-
tive or positive values, respectively. Porphyry
deposits were associated with relatively rapid
trench retreat compared to the random points
(mean values: positive −2.20 cm yr−1, random
−0.915 cm yr−1; Figs. 4D and 5D), suggesting
that they are unlikely to form during periods of
trench advance. Rapid trench retreat relative to
the underlying asthenosphere, however, could
encourage more effective rejuvenation of the
mantle wedge, when compared to a less mobile
subduction system. Furthermore, along with two
of the other features with high importance val-
ues—seafloor spreading rate for the subducting
crust and subduction convergence rate—trench
velocity has previously been found to be strongly
related to the dip angle of the subducting slab
(Mather et al., 2023). Studies of recent porphyry
copper systems from different regions of the
globe have indicated that slab dip angle may be
related to deposit formation, though the nature of
this relationship remains unclear. In the circum-
Pacific region, many giant porphyry deposits
coincide with flat-slab subduction (Cooke et al.,
2005). Conversely, most of the large Himalayan
Gangdese belt deposits formed above an area of
relatively steep subduction (Sun et al., 2021),
though some smaller deposits have also been
found above a flat slab segment (Zhou and Wang,
2023). The different nature of the subducting
crust in the Himalayan and circum-Pacific set-
tings (continental versus oceanic, respectively)
has been suggested as a possible explanation
for this discrepancy in slab dip angle (Sun et al.,
2021). Furthermore, spatial (Sun et al., 2021) or
temporal (Zhou et al., 2021) changes in slab dip
could lead to slab tearing, which might facilitate
extensive porphyry deposit formation (Logan
and Mihalynuk, 2014).
The spreading rate at the mid-ocean ridge
where the subducting oceanic crust was ­ created
706 Geological Society of America Bulletin, v. 137, no. 1/2
Downloaded from http://pubs.geoscienceworld.org/gsa/gsabulletin/article-pdf/137/1-2/702/7082726/b37614.1.pdf
by Australian National University user
Copper prospectivity in the American Cordillera
A B
C D
Figure 4. Boxplots for the six
most important features iden-
tified in Figure 3, showing the
distribution of values for posi-
tive/deposit (left) and random
(right) locations. Distributions
are shown for North America
(NA), South America (SA), and
both regions (All).
E F
exhibited a similarly high importance value
(mean values: positive 118 km m.y.−1, random
87.2 km m.y.−1; Fig. 4E). Fast seafloor spreading
rates (>∼80 km m.y.−1) produced higher pro-
spectivity values (Fig. 5E); such rapid spread-
ing is associated with high rates of volatile
extraction from the underlying mantle, result-
ing in increased sequestration of materials such
as water and (CO2) in the oceanic lithosphere
(Keller et al., 2017). These volatiles might later
be released into the mantle wedge during sub-
duction, facilitating magmatism and the devel-
opment of porphyry systems (e.g., Diaz-Rodri-
guez et al., 2021; Richards, 2003).
The sixth parameter identified was the cumu-
lative volume density (dimensionally equivalent
to thickness) of subducted carbonate sediment.
Porphyry deposits were generally associated
with greater quantities of subducted carbonates
(mean values: positive 340 m, random 207 m;
Figs. 4F and 5F), reinforcing the evidence for
a link between subducting carbon (in this case,
in the form of carbonate, CO3
2− ) and porphyry
deposit formation. Meanwhile, known non-
deposits were instead found to exhibit dramati-
cally high values for this parameter, particularly
in South America (mean values: positive 407 m,
random 215 m, negative 1726 m); this apparent
discrepancy is likely due to the young age of
most of these rocks, as previously observed by
Diaz-Rodriguez et al. (2021).
To highlight any potential differences
between porphyry deposit formation in North
and South America, feature importance and
partial dependence values were also extracted
from the PU classifiers trained on the datasets
restricted to these regions (Fig. 3). While con-
siderable similarities were observed between
the full and North American models, the only
features common to the top six for all three
models were the mean crustal thickness and
distance to the trench. Rank correlation (Ken-
dall’s τ) of feature importances for the differ-
ent models was also performed, with values of
τ
=
0.64 (all deposits/North America), τ
=
044
(all deposits/South America), and τ
=
0.20
(North America/South America); the final value
indicates significant differences between the
important features for North and South Amer-
ica. These results suggest that while porphyry
copper provinces may be formed in a variety
of tectonic settings, sufficiently thick crust, a
location far enough inboard of the subduction
zone, and a plentiful supply of volatiles such as
(CO2) and CO3
2− are crucial to the formation of
deposits in all scenarios.
Model Performance
Most conventional classifier performance
metrics (accuracy, F1 score, etc.) require a
well-balanced testing dataset, containing an
approximately equal number of positive and
negative observations. Though a PU classifier
can be trained on partially labeled data, the
use of performance metrics calculated using
Geological Society of America Bulletin, v. 137, no. 1/2 707
Downloaded from http://pubs.geoscienceworld.org/gsa/gsabulletin/article-pdf/137/1-2/702/7082726/b37614.1.pdf
by Australian National University user
Alfonso et al.
A B
C D
E F
Figure 5. Partial dependence plots for the six most important features identified in Figure 3,
showing the mean impact of each variable on predicted deposit probability (prospectivity).
Vertical ticks on the x-axis represent deciles in the training data.
only positive and unlabeled observations can
be highly problematic without prior knowledge
of the class proportions in the unlabeled data
(Ramola et al., 2019). Therefore, in addition
to the positive data points used to train the PU
models, the negative (non-deposit) observa-
tions from the same dataset (Diaz-Rodriguez
et al., 2021) were used in the evaluation of
these models.
Precision-recall and receiver operating
characteristic (ROC) curves calculated for
the PU and SVM models are shown in Fig-
ure 6. These curves were generated for mod-
els trained on the full test dataset (Figs. 6A
and 6B), North America only (Figs. 6C and
6D), and South America only (Figs. 6E and
6F). The two models performed equally well
on test data constrained to North America
(Figs. 6C and 6D), with very high average
precision (AP) and ROC area-under-the-curve
(AUC) values (PU ≈ SVM ≈ 1.0). However,
the PU model showed relatively poor perfor-
mance for South America, including both the
ROC AUC (PU
=
0.80, SVM
=
0.99) and AP
values (PU
=
0.72, SVM
=
0.97). Similarly,
when applied to the combined dataset, the PU
approach resulted in slightly inferior perfor-
mance, as shown by ROC AUC (PU
=
0.94,
SVM
=
0.99) and AP (PU
=
0.94,
SVM
=
0.97). The variation in model perfor-
mance between different regions indicates that
while PU models are often capable of produc-
ing similar results to traditional methods, this
capability is not universal. Furthermore, the
metrics obtained for the SVM model were
nearly identical to those calculated by Diaz-
Rodriguez et al. (2021) for an SVM trained
using a significantly different plate model
(Müller et al., 2016) than that used here; this
outcome reflects the strong bias in the dataset
toward younger ages (more than 97% of obser-
vations were dated to after 100 Ma), when the
differences between the plate reconstructions
were far less pronounced.
CONCLUSIONS
While previous studies have established the
capability of a PU classifier to match or exceed
the performance of a traditional machine learn-
ing model when applied to regional-scale min-
eral prospectivity mapping (e.g., Xiong and Zuo,
2021; Zhang et al., 2021), these results indicate
that this capability can also be extended to much
larger scales. Training a model without nega-
tively labeled data avoids making any assump-
tions about non-prospective locations and times,
unlike most methods used to generate these neg-
ative observations when confirmed non-deposit
datasets are unavailable (e.g., Carranza et al.,
2008; Diaz-Rodriguez et al., 2021).
The incorporation of additional predic-
tors derived from the overriding plate into the
machine learning models allowed the identifi-
cation of crustal thickness as the feature most
strongly associated with porphyry copper
deposits, providing further statistical confirma-
tion of the relationship between thick (gener-
ally continental) arc crust and these deposits, as
outlined by Lee and Tang (2020). Additionally,
the important subduction-related features seen
here, including convergence rate and subduc-
tion of large quantities of carbon, were simi-
lar to those identified by Diaz-Rodriguez et al.
(2021), despite the use of a significantly differ-
ent plate model. These results further reinforce
the conclusion that these parameters strongly
influence the formation of porphyry copper
deposits.
While the modeled high-prospectivity zones
largely correspond to areas of confirmed deposit
formation contained within the training data, a
number of these zones in locations and times
without any known deposits could present
708 Geological Society of America Bulletin, v. 137, no. 1/2
Downloaded from http://pubs.geoscienceworld.org/gsa/gsabulletin/article-pdf/137/1-2/702/7082726/b37614.1.pdf
by Australian National University user
Copper prospectivity in the American Cordillera
A B
C D
Figure 6. (A, C, E) Precision-
recall and (B, D, F) receiver
operating characteristic (ROC)
curves for the positive-unla-
beled (PU) and support vec-
tor machine (SVM) classifiers,
trained on the three datasets.
All—both regions; NA—North
America; SA—South America.
Also shown are average preci-
sion (AP) and ROC area-under-
the-curve (AUC) values.
E F
promising new areas for exploration. ­ However,
as shown by the poor results for the South
American deposits, considerable care must be
taken when using this approach. Ideally, before
applying a PU model to a given region and time
period, the algorithm’s performance relative to
traditional methods should be evaluated using a
similar setting for which more data is available.
Nevertheless, were the methods developed here
to be applied to data-poor regions and time peri-
ods, many more potential targets for porphyry
copper exploration might be identified around
the globe.
ACKNOWLEDGMENTS
This work was undertaken through Spatio-Tempo-
ral Exploration for Resources, a joint project between
the University of Sydney School of Geosciences (Syd-
ney, Australia) and BHP Group Limited (Melbourne,
Australia).
REFERENCES CITED
Athy, L.F., 1930, Density, porosity, and compaction of sedi-
mentary rocks: AAPG Bulletin, v. 14, no. 1, p. 1–24.
Bekker, J., and Davis, J., 2020, Learning from positive and un-
labeled data: A survey: Machine Learning, v. 109, no. 4,
p. 719–760, https://doi .org /10 .1007 /s10994 -020-05877-5.
Butterworth, N., Steinberg, D., Müller, R.D., Williams, S.,
Merdith, A.S., and Hardy, S., 2016, Tectonic environ-
ments of South American porphyry copper magmatism
through time revealed by spatiotemporal data mining:
Tectonics, v. 35, no. 12, p. 2847–2862, https://doi .org
/10 .1002 /2016TC004289.
Geological Society of America Bulletin, v. 137, no. 1/2 709
Downloaded from http://pubs.geoscienceworld.org/gsa/gsabulletin/article-pdf/137/1-2/702/7082726/b37614.1.pdf
by Australian National University user
Alfonso et al.
Cao, W., Zahirovic, S., Flament, N., Williams, S., Golonka,
J., and Müller, R.D., 2017, Improving global paleoge-
ography since the late Paleozoic using paleobiology:
Biogeosciences, v. 14, no. 23, p. 5425–5439, https://doi
.org /10 .5194 /bg-14-5425-2017.
Capitanio, F.A., 2013, Lithospheric-age control on the migra-
tions of oceanic convergent margins: Tectonophysics,
v. 593, p. 193–200, https://doi .org /10 .1016 /j .tecto .2013
.03 .003.
Capitanio, F.A., Stegman, D.R., Moresi, L.N., and Sharples,
W., 2010, Upper plate controls on deep subduction,
trench migrations and deformations at convergent
margins: Tectonophysics, v. 483, no. 1–2, p. 80–92,
https://doi .org /10 .1016 /j .tecto .2009 .08 .020.
Carranza, E.J.M., and Laborte, A.G., 2015, Random forest
predictive modeling of mineral prospectivity with small
number of prospects and data with missing values in
Abra (Philippines): Computers & Geosciences, v. 74,
p. 60–70, https://doi .org /10 .1016 /j .cageo .2014 .10 .004.
Carranza, E.J.M., Hale, M., and Faassen, C., 2008, Selection
of coherent deposit-type locations and their application
in data-driven mineral prospectivity mapping: Ore Ge-
ology Reviews, v. 33, no. 3–4, p. 536–558, https://doi
.org /10 .1016 /j .oregeorev .2007 .07 .001.
Chen, T., and Guestrin, C., 2016, XGBoost: A scalable tree
boosting system, in Proceedings of the 22nd ACM SIG-
KDD International Conference on Knowledge Discov-
ery and Data Mining, San Francisco, California, USA:
New York, Association for Computing Machinery,
p. 785–794, https://doi .org /10 .1145 /2939672 .2939785.
Clennett, E.J., Sigloch, K., Mihalynuk, M.G., Seton, M.,
Henderson, M.A., Hosseini, K., Mohammadzaheri, A.,
Johnston, S.T., and Müller, R.D., 2020, A quantitative
tomotectonic plate reconstruction of Western North
America and the Eastern Pacific Basin: Geochemistry,
Geophysics, Geosystems, v. 21, no. 8, https://doi .org /10
.1029 /2020GC009117.
Cooke, D.R., Hollings, P., and Walshe, J.L., 2005, Giant
porphyry deposits: Characteristics, distribution, and
tectonic controls: Economic Geology, v. 100, no. 5,
p. 801–818, https://doi .org /10 .2113 /gsecongeo .100
.5.801.
Department of Regional NSW, 2021, Critical Minerals and
High-Tech Metals Strategy: State of New South Wales
(Department of Regional NSW), Australia, https://www
.nsw .gov .au /regional-nsw /critical-minerals-and-high
-tech-metals-strategy (accessed March 2023).
Diaz-Rodriguez, J., Müller, R.D., and Chandra, R., 2021,
Predicting the emplacement of Cordilleran porphyry
copper systems using a spatio-temporal machine learn-
ing model: Ore Geology Reviews, v. 137, https://doi .org
/10 .1016 /j .oregeorev .2021 .104300.
Dickinson, W.R., 2004, Evolution of the North American
Cordillera: Annual Review of Earth and Planetary Sci-
ences, v. 32, no. 1, p. 13–45, https://doi .org /10 .1146
/annurev .earth .32 .101802 .120257.
Dickinson, W.R., 2008, Accretionary Mesozoic–Cenozoic
expansion of the Cordilleran continental margin in Cali-
fornia and adjacent Oregon: Geosphere, v. 4, p. 329–
353, https://doi .org /10 .1130 /GES00105 .1.
Dutkiewicz, A., Müller, R.D., Wang, X., O’Callaghan,
S., Cannon, J., and Wright, N.M., 2017, Predicting
sediment thickness on vanished ocean crust since
200 Ma: Geochemistry, Geophysics, Geosystems,
v. 18, no. 12, p. 4586–4603, https://doi .org /10 .1002
/2017GC007258.
Dutkiewicz, A., Müller, R.D., Cannon, J., Vaughan, S., and
Zahirovic, S., 2019, Sequestration and subduction of
deep-sea carbonate in the global ocean since the Early
Cretaceous: Geology, v. 47, p. 91–94, https://doi .org /10
.1130 /G45424 .1.
Engebretson, D.C., Cox, A., and Gordon, R.G., 1985, Rela-
tive Motions Between Oceanic and Continental Plates
in the Pacific Basin: Geological Society of America
Special Paper 206, 60 p., https://doi .org /10 .1130
/SPE206-p1.
Gehrels, G.E., McClelland, W.C., and Yokelson, I., 2017,
Reply to “Comment on ‘U-Pb and Hf isotope analysis
of detrital zircons from Mesozoic strata of the Gravina
Belt, southeast Alaska’ by Yokelson et al. (2015)”: Tec-
tonics, v. 36, no. 11, p. 2741–2743, https://doi .org /10
.1002 /2017TC004735.
Glen, R.A., Quinn, C.D., and Cooke, D.R., 2012, The Mac-
quarie Arc, Lachlan Orogen, New South Wales: Its evo-
lution, tectonic setting and mineral deposits: Episodes,
v. 35, no. 1, p. 177–186, https://doi .org /10 .18814 /epiiugs
/2012 /v35i1 /017.
Ingersoll, R.V., 2008, Subduction-related sedimentary basins
of the USA Cordillera, in Miall, A.D., ed., Sedimen-
tary Basins of the World: Amsterdam, Elsevier, v. 5,
p. 395–428.
Ingersoll, R.V., and Schweickert, R.A., 1986, A plate-tecton-
ic model for Late Jurassic ophiolite genesis, Nevadan
orogeny and forearc initiation, northern California: Tec-
tonics, v. 5, no. 6, p. 901–912, https://doi .org /10 .1029
/TC005i006p00901.
Keller, T., Katz, R.F., and Hirschmann, M.M., 2017, Volatiles
beneath mid-ocean ridges: Deep melting, channelised
transport, focusing, and metasomatism: Earth and Plan-
etary Science Letters, v. 464, p. 55–68, https://doi .org
/10 .1016 /j .epsl .2017 .02 .006.
Kreuzer, O.P., Miller, A.V.M., Peters, K.J., Payne, C., Wild-
man, C., Partington, G.A., Puccioni, E., McMahon,
M.E., and Etheridge, M.A., 2015, Comparing prospec-
tivity modelling results and past exploration data: A
case study of porphyry Cu-Au mineral systems in the
Macquarie Arc, Lachlan Fold Belt, New South Wales:
Ore: Geological Review (Dizhi Lunping), v. 71, p. 516–
544, https://doi .org /10 .1016 /j .oregeorev .2014 .09 .001.
LaMaskin, T.A., Rivas, J.A., Barbeau, D.L., Schwartz, J.J.,
Russell, J.A., and Chapman, A.D., 2022, A crucial geo-
logic test of Late Jurassic exotic collision versus en-
demic re-accretion in the Klamath Mountains Province,
western United States, with implications for the assem-
bly of western North America: Geological Society of
America Bulletin, v. 134, p. 965–988, https://doi .org
/10 .1130 /B35981 .1.
Lee, C.-T.A., and Tang, M., 2020, How to make porphyry
copper deposits: Earth and Planetary Science Letters,
v. 529, https://doi .org /10 .1016 /j .epsl .2019 .115868.
Lindsay, M.D., Piechocka, A.M., Jessell, M.W., Scalzo, R.,
Giraud, J., Pirot, G., and Cripps, E., 2022, Assessing the
impact of conceptual mineral systems uncertainty on
prospectivity predictions: Geoscience Frontiers, v. 13,
no. 6, https://doi .org /10 .1016 /j .gsf .2022 .101435.
Logan, J.M., and Mihalynuk, M.G., 2014, Tectonic controls
on early Mesozoic paired alkaline porphyry deposit
belts (Cu-Au Ag-Pt-Pd-Mo) within the Canadian Cor-
dillera: Economic Geology, v. 109, no. 4, p. 827–858,
https://doi .org /10 .2113 /econgeo .109 .4.827.
Loucks, R.R., 2014, Distinctive composition of copper-
ore-forming arc magmas: Australian Journal of Earth
Sciences, v. 61, no. 1, p. 5–16, https://doi .org /10 .1080
/08120099 .2013 .865676.
Lowey, G.W., 2017, Comment on “U-Pb and Hf isotope
analysis of detrital zircons from Mesozoic strata of
the Gravina Belt, southeast Alaska” by Yokelson
et al. (2015): Tectonics, v. 36, no. 11, p. 2736–2740,
https://doi .org /10 .1002 /2017TC004507.
Lowey, G.W., 2023, The good, the bad, and the ugly: Analy-
sis of three arguments in the ongoing debate concerning
the polarity of Mesozoic arcs along the western mar-
gin of North America: Geological Society of America
Bulletin, v. 135, p. 2591–2600, https://doi .org /10 .1130
/B36706 .1.
Mather, B.R., Muller, R.D., Alfonso, C.P., Seton, M., and
Wright, N.M., 2023, Kimberlite eruptions driven by
slab flux and subduction angle: Scientific Reports,
v. 13, no. 9216, https://doi .org /10 .1038 /s41598-023
-36250-w.
Meyer, B., Saltus, R., and Chulliat, A., 2017, EMAG2v3:
Earth Magnetic Anomaly Grid (2-arc-minute reso-
lution), Volume 2022: U.S. National Oceanic and
Atmospheric Administration, National Centers for
Environmental Information, https://www .ncei .noaa
.gov /products /earth-magnetic-model-anomaly-grid-2
(accessed April 2022).
Moores, E.M., 1998, Ophiolites, the Sierra Nevada, “Cor-
dilleria,” and orogeny along the Pacific and Caribbean
margins of North and South America: International Ge-
ology Review, v. 40, no. 1, p. 40–54, https://doi .org /10
.1080 /00206819809465197.
Mordelet, F., and Vert, J.P., 2014, A bagging SVM to learn
from positive and unlabeled examples: Pattern Rec-
ognition Letters, v. 37, p. 201–209, https://doi .org /10
.1016 /j .patrec .2013 .06 .010.
Müller, R.D., and Dutkiewicz, A., 2018, Oceanic crustal car-
bon cycle drives 26-million-year atmospheric carbon
dioxide periodicities: Science Advances, v. 4, no. 2,
https://doi .org /10 .1126 /sciadv .aaq0500.
Müller, R.D., Seton, M., Zahirovic, S., Williams, S.E., Mat-
thews, K.J., Wright, N.M., Shephard, G.E., Maloney,
K.T., Barnett-Moore, N., Hosseinpour, M., Bower,
D.J., and Cannon, J., 2016, Ocean basin evolution and
global-scale plate reorganization events since Pangea
breakup: Annual Review of Earth and Planetary Sci-
ences, v. 44, no. 1, p. 107–138, https://doi .org /10 .1146
/annurev-earth-060115-012211.
Müller, R.D., Zahirovic, S., Williams, S.E., Cannon, J., Se-
ton, M., Bower, D.J., Tetley, M.G., Heine, C., Le Bret-
on, E., Liu, S., Russell, S.H.J., Yang, T., Leonard, J.,
and Gurnis, M., 2019, A global plate model including
lithospheric deformation along major rifts and orogens
since the Triassic: Tectonics, v. 38, no. 6, p. 1884–1907,
https://doi .org /10 .1029 /2018TC005462.
Nykänen, V., Lahti, I., Niiranen, T., and Korhonen, K., 2015,
Receiver operating characteristics (ROC) as validation
tool for prospectivity models: A magmatic Ni-Cu case
study from the Central Lapland Greenstone Belt, North-
ern Finland: Ore Geology Reviews, v. 71, p. 853–860,
https://doi .org /10 .1016 /j .oregeorev .2014 .09 .007.
Park, J.-W., Campbell, I.H., Chiaradia, M., Hao, H., and Lee,
C.-T., 2021, Crustal magmatic controls on the forma-
tion of porphyry copper deposits: Nature Reviews Earth
& Environment, v. 2, no. 8, p. 542–557, https://doi .org
/10 .1038 /s43017-021-00182-8.
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V.,
Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P.,
Weiss, R., Dubourg, V., Vanderplas, J., Passos, A.,
Cournapeau, D., Brucher, M., Perrot, M., and Duch-
esnay, É., 2011, Scikit-learn: Machine learning in Py-
thon: Journal of Machine Learning Research, v. 12, no.
85, p. 2825–2830.
Ramola, R., Jain, S., and Radivojac, P., 2019, Estimating
classification accuracy in positive-unlabeled learn-
ing: Characterization and correction strategies, in Alt-
man, R.B., Dunker, A.K., Hunter, L., Ritchie, M.D.,
Murray, T., and Klein, T.E., eds., Pacific Symposium
on Biocomputing 2019: Singapore, World Scien-
tific Publishing, p. 124–135, https://doi .org /10 .1142
/9789813279827_0012.
Richards, J.P., 2003, Tectono-magmatic precursors for por-
phyry Cu-(Mo-Au): Deposit formation: Economic
Geology, v. 98, no. 8, p. 1515–1533, https://doi .org /10
.2113 /gsecongeo .98 .8.1515.
Richards, J.P., 2022, Porphyry copper deposit formation in
arcs: What are the odds?: Geosphere, v. 18, p. 130–155,
https://doi .org /10 .1130 /GES02086 .1.
Schellart, W.P., 2008, Subduction zone trench migration:
Slab driven or overriding-plate-driven?: Physics of the
Earth and Planetary Interiors, v. 170, no. 1–2, p. 73–88,
https://doi .org /10 .1016 /j .pepi .2008 .07 .040.
Schellart, W.P., Freeman, J., Stegman, D.R., Moresi, L., and
May, D., 2007, Evolution and diversity of subduction
zones controlled by slab width: Nature, v. 446, p. 308–
311, https://doi .org /10 .1038 /nature05615.
Seton, M., Müller, R.D., Zahirovic, S., Gaina, C.,
­ Torsvik, T., Shephard, G., Talsma, A., Gurnis, M.,
­ Turner, M., Maus, S., and Chandler, M., 2012, Global
continental and ocean basin reconstructions since
200 Ma: Earth-Science Reviews, v. 113, no. 3–4,
p. 212–270, https://doi .org /10 .1016 /j .earscirev .2012
.03 .002.
Sigloch, K., 2011, Mantle provinces under North America
from multifrequency P wave tomography: Geochem-
istry, Geophysics, Geosystems, v. 12, no. 2, https://doi
.org /10 .1029 /2010GC003421.
Sigloch, K., and Mihalynuk, M.G., 2013, Intra-oceanic
subduction shaped the assembly of Cordilleran North
America: Nature, v. 496, p. 50–56, https://doi .org /10
.1038 /nature12019.
Sigloch, K., and Mihalynuk, M.G., 2017, Mantle and geo-
logical evidence for a Late Jurassic–Cretaceous suture
spanning North America: Geological Society of Amer-
ica Bulletin, v. 129, p. 1489–1520, https://doi .org /10
.1130 /B31529 .1.
710 Geological Society of America Bulletin, v. 137, no. 1/2
Downloaded from http://pubs.geoscienceworld.org/gsa/gsabulletin/article-pdf/137/1-2/702/7082726/b37614.1.pdf
by Australian National University user
Copper prospectivity in the American Cordillera
Sigloch, K., McQuarrie, N., and Nolet, G., 2008, Two-stage
subduction history under North America inferred from
multiple-frequency tomography: Nature Geoscience,
v. 1, no. 7, p. 458–462, https://doi .org /10 .1038 /ngeo231.
Sillitoe, R.H., 2010, Porphyry copper systems: Economic
Geology, v. 105, no. 1, p. 3–41, https://doi .org /10 .2113
/gsecongeo .105 .1.3.
Sillitoe, R.H., 2012, Copper provinces, in Hedenquist, J.W.,
Harris, M., and Camus, F., eds., Geology and Genesis
of Major Copper Deposits and Districts of the World:
A Tribute to Richard H. Sillitoe: Society of Economic
Geologists Special Publication 16, p. 1–18.
Stegman, D.R., Farrington, R., Capitanio, F.A., and Schel-
lart, W.P., 2010, A regime diagram for subduction styles
from 3-D numerical models of free subduction: Tecto-
nophysics, v. 483, no. 1–2, p. 29–45, https://doi .org /10
.1016 /j .tecto .2009 .08 .041.
Sun, X., Lu, Y., Li, Q., and Li, R., 2021, A downgoing In-
dian lithosphere control on along-strike variability of
porphyry mineralization in the Gangdese Belt of south-
ern Tibet: Economic Geology, v. 116, no. 1, p. 29–46,
https://doi .org /10 .5382 /econgeo .4768.
Tetley, M.G., Williams, S.E., Gurnis, M., Flament, N.,
and Müller, R.D., 2019, Constraining absolute plate
motions since the Triassic: Journal of Geophysical
Research: Solid Earth, v. 124, no. 7, p. 7231–7258,
https://doi .org /10 .1029 /2019JB017442.
Tikoff, B., Housen, B.A., Maxson, J.A., Nelson, E.M., Tre-
vino, S., and Shipley, T.F., 2022, Hit-and-run model
for Cretaceous–Paleogene tectonism along the west-
ern margin of Laurentia, in Whitmeyer, S.J., Williams,
M.L., Kellett, D.A., and Tikoff, B., eds., Laurentia:
Turning Points in the Evolution of a Continent: Geo-
logical Society of America Memoir 220, p. 659–706,
https://doi .org /10 .1130 /2022 .1220(32).
Tosdal, R.M., and Richards, J.P., 2001, Magmatic and struc-
tural controls on the development of porphyry Cu ±
Mo ± Au deposits, in Richards, J.P., and Tosdal, R.M.,
eds., Structural Controls on Ore Genesis: Society of
Economic Geologists, Reviews in Economic Geology
14, p. 157–181.
Wilkinson, J.J., 2013, Triggers for the formation of porphyry
ore deposits in magmatic arcs: Nature Geoscience, v. 6,
no. 11, p. 917–925, https://doi .org /10 .1038 /ngeo1940.
Williams, S., Wright, N.M., Cannon, J., Flament, N., and
Müller, R.D., 2021, Reconstructing seafloor age dis-
tributions in lost ocean basins: Geoscience Frontiers,
v. 12, no. 2, p. 769–780, https://doi .org /10 .1016 /j .gsf
.2020 .06 .004.
Xiong, Y., and Zuo, R., 2021, A positive and unlabeled learn-
ing algorithm for mineral prospectivity mapping: Com-
puters & Geosciences, v. 147, https://doi .org /10 .1016 /j
.cageo .2020 .104667.
Yokelson, I., Gehrels, G.E., Pecha, M., Giesler, D., White,
C., and McClelland, W.C., 2015, U-Pb and Hf isotope
analysis of detrital zircons from Mesozoic strata of the
Gravina belt, southeast Alaska: Tectonics, v. 34, no. 10,
p. 2052–2066, https://doi .org /10 .1002 /2015TC003955.
Zhang, Z., Wang, G., Liu, C., Cheng, L., and Sha, D., 2021,
Bagging-based positive-unlabeled learning algorithm
with Bayesian hyperparameter optimization for three-
dimensional mineral potential mapping: Computers &
Geosciences, v. 154, https://doi .org /10 .1016 /j .cageo
.2021 .104817.
Zhou, Q., and Wang, R., 2023, Shallow subduction of Indian
slab and tectono-magmatic control on post-collisional
porphyry mineralization in southeastern Tibet: Ore
Geology Reviews, v. 155, https://doi .org /10 .1016 /j
.oregeorev .2023 .105360.
Zhou, Y., Cheng, Q., Liu, Y., Zhu, P., Wu, G., Zhang, Z., and
Yang, J., 2021, Singularity analysis of igneous zircon
U-Pb age and Hf isotopic record in the Zhongdian arc,
northwest Yunnan, China: Implications for Indosinian
magmatic flare-up and the formation of porphyry cop-
per deposits: Ore Geology Reviews, v. 139, https://doi
.org /10 .1016 /j .oregeorev .2021 .104476.
Zuo, R., and Carranza, E.J.M., 2011, Support vector ma-
chine: A tool for mapping mineral prospectivity: Com-
puters & Geosciences, v. 37, no. 12, p. 1967–1975,
https://doi .org /10 .1016 /j .cageo .2010 .09 .014.
Science Editor: Mihai Ducea
Associate Editor: Santiago Tassara
Manuscript Received 14 February 2024
Revised Manuscript Received 21 May 2024
Manuscript Accepted 16 July 2024
Geological Society of America Bulletin, v. 137, no. 1/2 711
Downloaded from http://pubs.geoscienceworld.org/gsa/gsabulletin/article-pdf/137/1-2/702/7082726/b37614.1.pdf
by Australian National University user
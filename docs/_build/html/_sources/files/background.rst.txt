=================
Background theory
=================

To reduce the burden of resource-intensive human and animal clinical trials for **acute ischaemic stroke**, *in-silico*
trials are sought as an alternative as part of the GEMINI project. These simulated trials - where brain meshes are
altered slightly to represent different members of the population (ages, sexes, &c.) - will enable predictions of
patient outcome as a function of treatments such as thrombectomy and thrombolysis. This will help inform clinical
decisions, like whether to carry out the treatment for a particular patient given that there are some associated risks.

One aspect of these simulations involves an organ-scale micro-circulation model of the human brain for estimating
perfusion of blood. An interrupted perfusion, i.e. when a clot occludes a blood vessel as it happens during a stroke,
reduces or suppresses the oxygen delivery to the cells. This is tightly linked to the health of the tissue, which is a
direct indicator of patient functional outcome.
This model is an essential part of simulating the brain mechanics.

The goal is to see how a cohort of virtual patients respond to a particular treatment. That is to run a clinical trial
to assess response to thrombolysis and thrombosis, the two main treatments of strokes, but virtually.

--------------------------------------------------
A Microscopic Circulation Model of the Human Brain
--------------------------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A Porous Continuum with Three Compartments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This model helps to predict the blood perfusion levels at different spatial points. The 1D blood flow model is coupled
with the cerebral perfusion model; from main arteries to the pial surface. The effect of blockages in the 1D blood
vessel network on distribution of blood on the pial surface is coupled with the response of perfusion in the brain.
Then there are additional factors depending on perfusion: oxygen/nutrient delivery, tissue damage, and oedema/swelling
(mechanical deformation of tissue) which are models also in development. The approach is modular in this respect.

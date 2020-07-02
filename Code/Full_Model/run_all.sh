#!/bin/bash

for i in {11..15}; do \
    python FullModel.py $i lowFix noRegrowth linear NormPop alphaStd; \
    tar -zcvf low_seed{$i}.tar.gz Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_lowFix_linear_NormPop_alphaStd_seed{$i}/
    rm -r Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_lowFix_linear_NormPop_alphaStd_seed{$i}/
done
for i in {11..15}; do \
    python FullModel.py $i lowFix withRegrowth linear NormPop alphaStd; \
    tar -zcvf lowwith_seed{$i}.tar.gz Figs_May21/FullModel_grid50_gH17e-3_withRegrowth_lowFix_linear_NormPop_alphaStd_seed{$i}/
    rm -r Figs_May21/FullModel_grid50_gH17e-3_withRegrowth_lowFix_linear_NormPop_alphaStd_seed{$i}/
done
for i in {11..15}; do \
    python FullModel.py $i highFix withRegrowth linear NormPop alphaStd; \
    tar -zcvf with_seed{$i}.tar.gz Figs_May21/FullModel_grid50_gH17e-3_withRegrowth_highFix_linear_NormPop_alphaStd_seed{$i}/
    rm -r Figs_May21/FullModel_grid50_gH17e-3_withRegrowth_highFix_linear_NormPop_alphaStd_seed{$i}/
done
for i in {11..15}; do \
    python FullModel.py $i highFix noRegrowth linear NormPop alphaResource; \
    tar -zcvf resource_seed{$i}.tar.gz Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_high_seFix_linear_NormPop_alphaResource_seed{$i}/
    rm -r Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaResource_seed{$i}/
done
for i in {11..15}; do \
    python FullModel.py $i highFix noRegrowth linear NormPop alphaDeterministic; \
    tar -zcvf Det_seed{$i}.tar.gz Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaDeterministic_seed{$i}/
    rm -r Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaDeterministic_seed{$i}/
done
for i in {11..15}; do \
    python FullModel.py $i highFix noRegrowth linear NormPop alphaHopping; \
    tar -zcvf Hoü_seed{$i}.tar.gz Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaHopping_seed{$i}/
    rm -r Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaHopping_seed{$i}/
done
for i in {11..15}; do \
    python FullModel.py $i highFix noRegrowth linear LessResPop alphaStd; \
    tar -zcvf LessResPop_seed{$i}.tar.gz Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_LessResPop_alphaStd_seed{$i}/
    rm -r Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_LessResPop_alphaStd_seed{$i}/
done
for i in {11..15}; do \
    python FullModel.py $i highFix noRegrowth logistic NormPop alphaStd; \
    tar -zcvf logistic_seed{$i}.tar.gz Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_highFix_logistic_NormPop_alphaStd_seed{$i}/
    rm -r Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_highFix_logistic_NormPop_alphaStd_seed{$i}/
done
for i in {11..15}; do \
    python FullModel.py $i highFix noRegrowth delayed NormPop alphaStd; \
    tar -zcvf delayed_seed{$i}.tar.gz Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_highFix_delayed_NormPop_alphaStd_seed{$i}/
    rm -r Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_highFix_delayed_NormPop_alphaStd_seed{$i}/
done
for i in {11..15}; do \
    python FullModel.py $i highFix noRegrowth careful NormPop alphaStd; \
    tar -zcvf careful_seed{$i}.tar.gz Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_highFix_careful_NormPop_alphaStd_seed{$i}/
    rm -r Figs_May21/FullModel_grid50_gH17e-3_noRegrowth_highFix_careful_NormPop_alphaStd_seed{$i}/
done
for i in {11..15}; do \
    python FullModel_largeRad.py $i highFix noRegrowth linear NormPop alphaStd; \
    tar -zcvf largeRad_seed{$i}.tar.gz Figs_May21_largeRad/FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaStd_seed{$i}/
    rm -r Figs_May21_largeRad/FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaStd_seed{$i}/
done
for i in {11..15}; do \
    python FullModel_smallRad.py $i highFix noRegrowth linear NormPop alphaStd; \
    tar -zcvf smallRad_seed{$i}.tar.gz Figs_May21_smallRad/FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaStd_seed{$i}/
    rm -r Figs_May21_smallRad/FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaStd_seed{$i}/
done
for i in {11..15}; do \
    python FullModel_largeTReq.py $i highFix noRegrowth linear NormPop alphaStd; \
    tar -zcvf largeTReq_seed{$i}.tar.gz Figs_May21_largeTReq/FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaStd_seed{$i}/
    rm -r Figs_May21_largeTReq/FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaStd_seed{$i}/
done


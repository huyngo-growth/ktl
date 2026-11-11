clear all
set more off
set linesize 160
capture ssc install winsor2
capture ssc install estout
log using "result.log", replace text
quietly import delimited "raw_data.csv", clear encoding("utf-8") varnames(1)
gen ATO = revenue / total_assets
gen TPR = trade_payables / total_assets
gen LIQ = current_assets / current_liab
gen SIZ = ln(total_assets)
gen LEV = (short_debt + long_debt) / total_assets
gen CSH = cash / total_assets
gen DLI = (LIQ >= 1)
encode ticker, gen(firm)
quietly xtset firm year
quietly winsor2 ATO TPR LIQ SIZ LEV CSH, cuts(1 99) replace
quietly summarize TPR, meanonly
gen TPRc = TPR - r(mean)
quietly summarize LIQ, meanonly
gen LIQc = LIQ - r(mean)
gen TLI = TPRc * LIQc
gen TLD = TPRc * DLI
global control "SIZ LEV CSH"
xtset firm year
xtdescribe
bysort firm: gen Ti = _N
egen tag = tag(firm)
tab Ti if tag == 1
xtsum ATO TPR LIQ SIZ LEV CSH
eststo clear
eststo POLS1: regress ATO TPRc LIQc $control i.year, vce(cluster firm)
xtreg ATO TPRc LIQc $control i.year, fe
estimates store fe1
xtreg ATO TPRc LIQc $control i.year, re
estimates store re1
xttest0
hausman fe1 re1, sigmamore
bysort firm: egen mn_TPRc = mean(TPRc)
bysort firm: egen mn_LIQc = mean(LIQc)
bysort firm: egen mn_SIZ = mean(SIZ)
bysort firm: egen mn_LEV = mean(LEV)
bysort firm: egen mn_CSH = mean(CSH)
quietly xtreg ATO TPRc LIQc $control mn_TPRc mn_LIQc mn_SIZ mn_LEV mn_CSH i.year, re vce(cluster firm)
testparm mn_TPRc mn_LIQc mn_SIZ mn_LEV mn_CSH
eststo FEM1: xtreg ATO TPRc LIQc $control i.year, fe vce(cluster firm)
eststo REM1: xtreg ATO TPRc LIQc $control i.year, re vce(cluster firm)
quietly xtreg ATO TPRc LIQc $control TLI i.year, fe
display "F test u_i = 0: F(" e(df_a) ", " e(df_r) ") = " %9.2f e(F_f) "   p = " %6.4f Ftail(e(df_a), e(df_r), e(F_f))
estimates store fe2
quietly xtreg ATO TPRc LIQc $control TLI i.year, re
estimates store re2
xttest0
hausman fe2 re2, sigmamore
bysort firm: egen mn_TLI = mean(TLI)
quietly xtreg ATO TPRc LIQc $control TLI mn_TPRc mn_LIQc mn_SIZ mn_LEV mn_CSH mn_TLI i.year, re vce(cluster firm)
testparm mn_TPRc mn_LIQc mn_SIZ mn_LEV mn_CSH mn_TLI
quietly xtreg ATO TPRc DLI $control TLD i.year, fe
display "F test u_i = 0: F(" e(df_a) ", " e(df_r) ") = " %9.2f e(F_f) "   p = " %6.4f Ftail(e(df_a), e(df_r), e(F_f))
estimates store fe3
quietly xtreg ATO TPRc DLI $control TLD i.year, re
estimates store re3
xttest0
hausman fe3 re3, sigmamore
bysort firm: egen mn_DLI = mean(DLI)
bysort firm: egen mn_TLD = mean(TLD)
quietly xtreg ATO TPRc DLI $control TLD mn_TPRc mn_DLI mn_SIZ mn_LEV mn_CSH mn_TLD i.year, re vce(cluster firm)
testparm mn_TPRc mn_DLI mn_SIZ mn_LEV mn_CSH mn_TLD
eststo POLS2: quietly regress ATO c.TPRc##c.LIQc $control i.year, vce(cluster firm)
eststo FEM2: quietly xtreg ATO c.TPRc##c.LIQc $control i.year, fe vce(cluster firm)
eststo REM2: quietly xtreg ATO c.TPRc##c.LIQc $control i.year, re vce(cluster firm)
eststo POLS3: quietly regress ATO c.TPRc##i.DLI $control i.year, vce(cluster firm)
eststo FEM3: quietly xtreg ATO c.TPRc##i.DLI $control i.year, fe vce(cluster firm)
eststo REM3: quietly xtreg ATO c.TPRc##i.DLI $control i.year, re vce(cluster firm)
esttab POLS1 FEM1 REM1 POLS2 FEM2 REM2 POLS3 FEM3 REM3, b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) drop(*.year) nobaselevels nogaps ///
    order(TPRc LIQc c.TPRc#c.LIQc 1.DLI 1.DLI#c.TPRc SIZ LEV CSH) scalars("N" "r2 R-sq" "r2_w R-sq within" "r2_o R-sq overall") ///
    mtitles("Pooled 1" "FEM 1" "REM 1" "Pooled 2" "FEM 2" "REM 2" "Pooled 3" "FEM 3" "REM 3")
bysort firm: egen bar_ATO = mean(ATO)
gen w_ATO = ATO - bar_ATO
bysort firm: egen bar_TPRc = mean(TPRc)
gen w_TPRc = TPRc - bar_TPRc
bysort firm: egen bar_LIQc = mean(LIQc)
gen w_LIQc = LIQc - bar_LIQc
bysort firm: egen bar_SIZ = mean(SIZ)
gen w_SIZ = SIZ - bar_SIZ
bysort firm: egen bar_LEV = mean(LEV)
gen w_LEV = LEV - bar_LEV
bysort firm: egen bar_CSH = mean(CSH)
gen w_CSH = CSH - bar_CSH
regress w_ATO w_TPRc w_LIQc w_SIZ w_LEV w_CSH, noconstant
xtreg ATO TPRc LIQc $control, fe
areg ATO TPRc LIQc $control, absorb(firm)
encode icb2, gen(nganh)
xtreg ATO TPRc LIQc $control i.year i.nganh, fe
list ticker year ATO bar_ATO w_ATO TPRc bar_TPRc w_TPRc if ticker == "VNM", clean
log close

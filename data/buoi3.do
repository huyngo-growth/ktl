clear all
set more off
set linesize 160
capture ssc install winsor2
capture ssc install estout
log using "result.log", replace text
quietly import delimited "raw_data.csv", clear encoding("utf-8") varnames(1)
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
tab DLI
tabstat ATO TPR LIQ SIZ LEV CSH, by(DLI) stat(n mean sd) format(%9.4f)
ttest ATO, by(DLI)
ttest TPR, by(DLI)
ttest SIZ, by(DLI)
ttest LEV, by(DLI)
ttest CSH, by(DLI)
eststo clear
eststo M0: regress ATO TPRc, vce(cluster firm)
eststo M1: regress ATO TPRc LIQc $control i.year, vce(cluster firm)
testparm i.year
quietly regress ATO TPRc LIQc $control
estat vif
eststo M2: regress ATO c.TPRc##c.LIQc $control i.year, vce(cluster firm)
quietly regress ATO TPRc LIQc $control TLI
estat vif
quietly summarize LIQ, meanonly
local mL = r(mean)
_pctile LIQ, p(25 50 75)
local c1 = r(r1) - `mL'
local c2 = r(r2) - `mL'
local c3 = r(r3) - `mL'
estimates restore M2
lincom TPRc + `c1'*c.TPRc#c.LIQc
lincom TPRc + `c2'*c.TPRc#c.LIQc
lincom TPRc + `c3'*c.TPRc#c.LIQc
eststo M3: regress ATO c.TPRc##i.DLI $control i.year, vce(cluster firm)
quietly regress ATO TPRc DLI $control TLD
estat vif
estimates restore M3
lincom TPRc + 1.DLI#c.TPRc
eststo S0: regress ATO TPRc LIQc $control i.year if DLI == 0, vce(cluster firm)
eststo S1: regress ATO TPRc LIQc $control i.year if DLI == 1, vce(cluster firm)
esttab M0 M1 M2 M3 S0 S1, b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) drop(*.year) nobaselevels r2 nogaps ///
    order(TPRc LIQc c.TPRc#c.LIQc 1.DLI 1.DLI#c.TPRc SIZ LEV CSH) mtitles("M0" "M1" "M2" "M3" "DLI=0" "DLI=1")
quietly summarize TPR, meanonly
local mT = r(mean)
_pctile TPR, p(10 90)
local lo = r(r1) - `mT'
local hi = r(r2) - `mT'
local st = (`hi' - `lo') / 20
quietly summarize SIZ, meanonly
local mS = r(mean)
quietly summarize LEV, meanonly
local mLEV = r(mean)
quietly summarize CSH, meanonly
local mCSH = r(mean)
estimates restore M2
quietly margins, at(TPRc = (`lo'(`st')`hi') LIQc = (`c1' `c2' `c3') SIZ = `mS' LEV = `mLEV' CSH = `mCSH' year = 2021)
marginsplot, noci recast(line) title("Tác động tương tác: phải trả người bán × thanh khoản") ///
    xtitle("Phải trả người bán / tổng tài sản (đã trừ trung bình)") ytitle("Vòng quay tổng tài sản dự báo") ///
    legend(order(1 "Thanh khoản thấp (p25)" 2 "Thanh khoản trung vị (p50)" 3 "Thanh khoản cao (p75)")) name(tuong_tac, replace)
graph export "tuong_tac.png", replace width(1600)
estimates restore M3
quietly margins DLI, at(TPRc = (`lo'(`st')`hi') SIZ = `mS' LEV = `mLEV' CSH = `mCSH' year = 2021)
marginsplot, noci recast(line) title("Tương tác với biến giả thanh khoản 0–1") ///
    xtitle("Phải trả người bán / tổng tài sản (đã trừ trung bình)") ytitle("Vòng quay tổng tài sản dự báo") ///
    legend(order(1 "DLI = 0: TSNH < nợ ngắn hạn" 2 "DLI = 1: TSNH ≥ nợ ngắn hạn")) name(tuong_tac_dummy, replace)
graph export "tuong_tac_dummy.png", replace width(1600)
log close

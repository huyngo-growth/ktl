clear all
set more off
set linesize 160
capture ssc install winsor2
capture ssc install heatplot
capture ssc install palettes
capture ssc install colrspace
log using "result.log", replace text
import delimited "raw_data.csv", clear encoding("utf-8") varnames(1)
count
tab template
keep if template == "nonfin"
drop if missing(revenue, total_assets, current_assets, current_liab, trade_payables)
drop if total_assets <= 0
drop if revenue <= 0
drop if current_liab <= 0
drop if current_assets < 0 | trade_payables < 0
count
gen ATO = revenue / total_assets
gen TPR = trade_payables / total_assets
gen LIQ = current_assets / current_liab
gen SIZ = ln(total_assets)
gen LEV = (short_debt + long_debt) / total_assets
gen CSH = cash / total_assets
label var ATO "Vòng quay tổng tài sản"
label var TPR "Phải trả người bán / tổng tài sản"
label var LIQ "Thanh khoản hiện hành"
label var SIZ "Quy mô: ln(tổng tài sản)"
label var LEV "Nợ vay / tổng tài sản"
label var CSH "Tiền / tổng tài sản"
encode ticker, gen(firm)
xtset firm year
tabstat ATO TPR LIQ SIZ LEV CSH, stat(n mean sd min p25 p50 p75 max skewness kurtosis) columns(statistics) format(%10.4f)
winsor2 ATO TPR LIQ SIZ LEV CSH, cuts(1 99) replace
tabstat ATO TPR LIQ SIZ LEV CSH, stat(n mean sd cv min p25 p50 p75 max skewness kurtosis) columns(statistics) format(%10.4f)
summarize ATO, detail
summarize TPR, detail
summarize LIQ, detail
summarize SIZ, detail
foreach v in ATO TPR LIQ SIZ LEV CSH {
    quietly summarize `v', detail
    local JB = r(N) / 6 * (r(skewness)^2 + (r(kurtosis) - 3)^2 / 4)
    display "`v'" _col(6) "Skew = " %7.3f r(skewness) "  Kurt = " %7.3f r(kurtosis) "  CV = " %7.3f r(sd)/abs(r(mean)) "  JB = " %12.2f `JB' "  p = " %6.4f chi2tail(2, `JB')
}
tabstat ATO TPR LIQ SIZ LEV CSH, by(year) stat(mean) format(%9.4f)
tab icb2, sort
histogram ATO, normal title("Vòng quay tổng tài sản") name(h1, replace)
histogram TPR, normal title("Phải trả người bán / tổng tài sản") name(h2, replace)
graph combine h1 h2, cols(2) name(hist, replace)
graph export "histogram.png", replace width(1600)
graph hbox ATO, name(b1, replace) title("Vòng quay tổng tài sản", size(medsmall))
graph hbox TPR, name(b2, replace) title("Phải trả người bán / tổng tài sản", size(medsmall))
graph hbox LIQ, name(b3, replace) title("Thanh khoản hiện hành", size(medsmall))
graph hbox SIZ, name(b4, replace) title("Quy mô: ln(tổng tài sản)", size(medsmall))
graph hbox LEV, name(b5, replace) title("Nợ vay / tổng tài sản", size(medsmall))
graph hbox CSH, name(b6, replace) title("Tiền / tổng tài sản", size(medsmall))
graph combine b1 b2 b3 b4 b5 b6, cols(2) title("Thống kê mô tả sau Winsor2 (p1–p99)") name(boxplot, replace)
graph export "boxplot.png", replace width(1600)
preserve
collapse (mean) TPR ATO, by(year)
replace TPR = TPR * 100
list, clean
twoway (connected TPR year, yaxis(1)) (connected ATO year, yaxis(2)), ///
    ytitle("Phải trả người bán / tổng tài sản (%)", axis(1)) ytitle("Vòng quay tổng tài sản (lần)", axis(2)) ///
    xtitle("Năm tài chính") xlabel(2021(1)2025) legend(order(1 "Phải trả / tổng tài sản" 2 "Hiệu quả hoạt động")) ///
    title("Phải trả người bán và hiệu quả hoạt động theo năm") name(theo_nam, replace)
graph export "theo_nam.png", replace width(1600)
restore
pwcorr ATO TPR LIQ SIZ LEV CSH, sig star(.05)
quietly correlate ATO TPR LIQ SIZ LEV CSH
matrix C = r(C)
heatplot C, values(format(%5.3f)) color(hcl diverging, intensity(.6)) legend(off) aspectratio(1) ///
    title("Ma trận hệ số tương quan Pearson") name(tuong_quan, replace)
graph export "tuong_quan.png", replace width(1600)
log close

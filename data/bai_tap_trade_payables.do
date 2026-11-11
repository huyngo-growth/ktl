*==== 0. Chuẩn bị: cài các lệnh bổ sung (chỉ cần chạy một lần)
* Cần Stata 14 trở lên để đọc được tiếng Việt (UTF-8).
* Các lệnh dưới đây tải gói từ SSC; máy cần kết nối Internet.
ssc install winsor2, replace
ssc install estout, replace
ssc install ftools, replace
ssc install reghdfe, replace
ssc install heatplot, replace
ssc install palettes, replace
ssc install colrspace, replace

*==== 1. Đọc dữ liệu đã lọc
* raw_data.csv đã được lọc sẵn từ dữ liệu gốc (chỉ doanh nghiệp phi tài chính; tổng tài sản, doanh thu, nợ ngắn hạn > 0; không có khoản mục âm hoặc thiếu).
* Đặt raw_data.csv cùng thư mục với file .do này, rồi sửa đường dẫn dưới đây cho đúng máy của bạn.
cd "C:\Users\Admin\Desktop\TradePayables_BaiTap"
import delimited "raw_data.csv", clear encoding("utf-8") varnames(1)
* Mỗi dòng là một công ty–năm FY2021–FY2025. Khoản mục tính bằng tỷ đồng; icb2: ngành ICB cấp 2.
describe
count
* Kết quả: 7.015 công ty–năm của 1.454 mã.

*==== 2. Phân bổ mẫu theo ngành (Bảng: Phân bổ mẫu phân tích theo ngành ICB cấp 2)
tab icb2, sort

*==== 3. Tạo biến và khai báo dữ liệu bảng
gen ATO = revenue / total_assets
gen TPR = trade_payables / total_assets
gen LIQ = current_assets / current_liab
gen SIZ = ln(total_assets)
gen LEV = (short_debt + long_debt) / total_assets
gen CSH = cash / total_assets
gen ROA = net_income / total_assets
gen DLI = (LIQ >= 1)
label var ATO "Vòng quay tổng tài sản"
label var TPR "Phải trả người bán / tổng tài sản"
label var LIQ "Thanh khoản hiện hành"
label var SIZ "ln(tổng tài sản, tỷ đồng)"
label var DLI "Dummy: 1 nếu LIQ >= 1"
label var LEV "Nợ vay / tổng tài sản"
label var CSH "Tiền / tổng tài sản"
label var ROA "Lợi nhuận sau thuế / tổng tài sản"
* Biến kiểm soát dùng chung cho mọi mô hình
global control "SIZ LEV CSH"
encode ticker, gen(firm)
xtset firm year
gen GRW = revenue / L.revenue - 1
label var GRW "Tăng trưởng doanh thu"
xtdescribe
tab DLI

*==== 4. Thống kê mô tả trước winsor hóa
tabstat ATO TPR LIQ SIZ LEV CSH, stat(n mean sd min p25 p50 p75 max skewness) columns(statistics) format(%10.4f)

*==== 5. Winsor2 p1–p99 (Bảng: Ngưỡng winsor)
foreach v in ATO TPR LIQ SIZ LEV CSH {
    _pctile `v', p(1 99)
    local p1 = r(r1)
    local p99 = r(r2)
    quietly count if `v' < `p1'
    local nlo = r(N)
    quietly count if `v' > `p99' & !missing(`v')
    display "`v'" _col(6) "p1 = " %9.4f `p1' "   p99 = " %9.4f `p99' "   dưới p1: `nlo'   trên p99: " r(N)
}
winsor2 ATO TPR LIQ SIZ LEV CSH ROA GRW, cuts(1 99) replace
* Stata tính phân vị theo định nghĩa hơi khác Python nên ngưỡng có thể lệch ở chữ số thập phân cuối.

*==== 6. Thống kê mô tả sau winsor hóa và biểu đồ mô tả
tabstat ATO TPR LIQ SIZ LEV CSH, stat(n mean sd min p25 p50 p75 max skewness) columns(statistics) format(%10.4f)
graph hbox ATO, name(g1, replace) title("Vòng quay tổng tài sản", size(medsmall))
graph hbox TPR, name(g2, replace) title("Phải trả người bán / tài sản", size(medsmall))
graph hbox LIQ, name(g3, replace) title("Thanh khoản hiện hành", size(medsmall))
graph hbox SIZ, name(g4, replace) title("Quy mô: ln(tổng tài sản)", size(medsmall))
graph combine g1 g2 g3 g4, cols(1) title("Thống kê mô tả sau Winsor2 (p1–p99)") name(mo_ta, replace)
graph export "bieu_do_thong_ke_mo_ta.png", replace width(1600)

*==== 7. Skewness, Kurtosis, CV và kiểm định Jarque–Bera
* summarize, detail trả về skewness = m3/m2^1.5 và kurtosis = m4/m2^2 (chuẩn = 3).
foreach v in ATO TPR LIQ SIZ {
    quietly summarize `v', detail
    local S  = r(skewness)
    local K  = r(kurtosis)
    local CV = r(sd) / abs(r(mean)) * 100
    local JB = r(N) / 6 * (`S'^2 + (`K' - 3)^2 / 4)
    display "`v'" _col(6) "Skew = " %6.3f `S' "  Kurt = " %7.3f `K' "  CV = " %8.3f `CV' "  JB = " %12.3f `JB' "  p = " %9.3g chi2tail(2, `JB')
}

*==== 8. Trung bình theo năm (biểu đồ hai trục)
preserve
collapse (mean) TPR ATO, by(year)
replace TPR = TPR * 100
list, clean
twoway (connected TPR year, yaxis(1)) (connected ATO year, yaxis(2)), ///
    ytitle("Phải trả người bán / tổng tài sản (%)", axis(1)) ytitle("Vòng quay tổng tài sản (lần)", axis(2)) ///
    xtitle("Năm tài chính") xlabel(2021(1)2025) ///
    legend(order(1 "Phải trả / tổng tài sản" 2 "Hiệu quả hoạt động")) ///
    title("Phải trả người bán và hiệu quả hoạt động theo năm") name(theo_nam, replace)
graph export "bieu_do_theo_nam.png", replace width(1600)
restore

*==== 9. Biến trừ trung bình, biến tương tác và ma trận tương quan Pearson
foreach v in TPR LIQ {
    quietly summarize `v', meanonly
    gen `v'c = `v' - r(mean)
}
gen TLI = TPRc * LIQc
gen TLD = TPRc * DLI
pwcorr ATO TPR LIQ SIZ LEV CSH DLI TLI TLD, star(.05)
quietly correlate ATO TPR LIQ SIZ LEV CSH DLI TLI TLD
matrix C = r(C)
heatplot C, values(format(%5.3f)) color(hcl diverging, intensity(.6)) legend(off) aspectratio(1) ///
    title("Ma trận hệ số tương quan Pearson") name(tuong_quan, replace)
graph export "bieu_do_tuong_quan.png", replace width(1600)

*==== 10. VIF cho ba mô hình
quietly regress ATO TPRc LIQc $control
estat vif
quietly regress ATO TPRc LIQc $control TLI
estat vif
quietly regress ATO TPRc DLI $control TLD
estat vif

*==== 11. Các mô hình OLS gộp (sai số chuẩn gom cụm theo mã)
* Viết tương tác dạng ## để lệnh margins hiểu: c.TPRc#c.LIQc chính là TLI, 1.DLI#c.TPRc chính là TLD.
* Stata tính p-value theo phân phối t với (số công ty − 1) bậc tự do nên có thể lệch rất nhỏ so với Python.
eststo clear
eststo M0: regress ATO TPRc, vce(cluster firm)
eststo M1: regress ATO TPRc LIQc $control i.year, vce(cluster firm)
eststo M2: regress ATO c.TPRc##c.LIQc $control i.year, vce(cluster firm)
predict RE2, residuals
eststo M3: regress ATO c.TPRc##i.DLI $control i.year, vce(cluster firm)
predict RE3, residuals
esttab M0 M1 M2 M3, b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) r2 ar2 nogaps
esttab M0 M1 M2 M3 using "ket_qua_OLS.rtf", replace b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) r2 ar2
* Jarque–Bera cho phần dư mô hình 2 và 3
foreach v in RE2 RE3 {
    quietly summarize `v', detail
    local JB = r(N) / 6 * (r(skewness)^2 + (r(kurtosis) - 3)^2 / 4)
    display "`v'" _col(6) "Skew = " %6.3f r(skewness) "  Kurt = " %7.3f r(kurtosis) "  JB = " %12.3f `JB' "  p = " %9.3g chi2tail(2, `JB')
}

*==== 12. Biểu đồ tương tác (marginsplot)
* Dự báo ATO theo TPR (đã trừ trung bình) từ p10 đến p90, giữ SIZ ở trung bình và năm gốc 2021.
quietly summarize TPR, meanonly
local mT = r(mean)
_pctile TPR, p(10 90)
local lo = r(r1) - `mT'
local hi = r(r2) - `mT'
local st = (`hi' - `lo') / 20
quietly summarize LIQ, meanonly
local mL = r(mean)
_pctile LIQ, p(25 50 75)
local q1 = r(r1) - `mL'
local q2 = r(r2) - `mL'
local q3 = r(r3) - `mL'
quietly summarize SIZ, meanonly
local mS = r(mean)
quietly summarize LEV, meanonly
local mLEV = r(mean)
quietly summarize CSH, meanonly
local mCSH = r(mean)
estimates restore M2
margins, at(TPRc = (`lo'(`st')`hi') LIQc = (`q1' `q2' `q3') SIZ = `mS' LEV = `mLEV' CSH = `mCSH' year = 2021)
marginsplot, noci recast(line) title("Tác động tương tác: phải trả người bán × thanh khoản") ///
    xtitle("Phải trả người bán / tổng tài sản (đã trừ trung bình)") ytitle("Vòng quay tổng tài sản dự báo") ///
    legend(order(1 "Thanh khoản thấp (p25)" 2 "Thanh khoản trung vị (p50)" 3 "Thanh khoản cao (p75)")) name(tuong_tac, replace)
graph export "bieu_do_tuong_tac.png", replace width(1600)
estimates restore M3
margins DLI, at(TPRc = (`lo'(`st')`hi') SIZ = `mS' LEV = `mLEV' CSH = `mCSH' year = 2021)
marginsplot, noci recast(line) title("Tương tác với biến giả thanh khoản 0–1") ///
    xtitle("Phải trả người bán / tổng tài sản (đã trừ trung bình)") ytitle("Vòng quay tổng tài sản dự báo") ///
    legend(order(1 "DLI = 0: TSNH < nợ ngắn hạn" 2 "DLI = 1: TSNH ≥ nợ ngắn hạn")) name(tuong_tac_dummy, replace)
graph export "bieu_do_tuong_tac_dummy.png", replace width(1600)

*==== 13. Kiểm tra độ bền: hiệu ứng cố định công ty và năm (FE)
* reghdfe tự loại công ty chỉ có 1 năm (singleton) nên còn 6.997 quan sát, giống Python.
eststo FE1: reghdfe ATO TPRc LIQc $control, absorb(firm year) vce(cluster firm)
eststo FE2: reghdfe ATO c.TPRc##c.LIQc $control, absorb(firm year) vce(cluster firm)
eststo FE3: reghdfe ATO c.TPRc##i.DLI $control, absorb(firm year) vce(cluster firm)
* Cách không cần reghdfe (giữ cả singleton, hệ số giống nhau):
* xtreg ATO TPRc LIQc SIZ i.year, fe vce(cluster firm)

*==== 14. Pooled OLS, FEM và REM cho ba mô hình
* regress = Pooled OLS; xtreg, fe = FEM (within); xtreg, re = REM (GLS). Cả ba gom cụm sai số chuẩn theo công ty.
eststo POLS1: regress ATO TPRc LIQc $control i.year, vce(cluster firm)
eststo FEM1:  xtreg   ATO TPRc LIQc $control i.year, fe vce(cluster firm)
eststo REM1:  xtreg   ATO TPRc LIQc $control i.year, re vce(cluster firm)
eststo POLS2: regress ATO c.TPRc##c.LIQc $control i.year, vce(cluster firm)
eststo FEM2:  xtreg   ATO c.TPRc##c.LIQc $control i.year, fe vce(cluster firm)
eststo REM2:  xtreg   ATO c.TPRc##c.LIQc $control i.year, re vce(cluster firm)
eststo POLS3: regress ATO c.TPRc##i.DLI $control i.year, vce(cluster firm)
eststo FEM3:  xtreg   ATO c.TPRc##i.DLI $control i.year, fe vce(cluster firm)
eststo REM3:  xtreg   ATO c.TPRc##i.DLI $control i.year, re vce(cluster firm)
esttab POLS1 FEM1 REM1 POLS2 FEM2 REM2 POLS3 FEM3 REM3, b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    drop(*.year, relax) scalars("r2_w R2 within" "r2_o R2 overall" "sigma_u sigma_u" "sigma_e sigma_e" "rho rho") r2 nogaps ///
    mtitles("Pooled 1" "FEM 1" "REM 1" "Pooled 2" "FEM 2" "REM 2" "Pooled 3" "FEM 3" "REM 3")
esttab POLS1 FEM1 REM1 POLS2 FEM2 REM2 POLS3 FEM3 REM3 using "pooled_fem_rem.rtf", replace b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) drop(*.year, relax) scalars("r2_w R2 within" "r2_o R2 overall" "rho rho") r2 ///
    mtitles("Pooled 1" "FEM 1" "REM 1" "Pooled 2" "FEM 2" "REM 2" "Pooled 3" "FEM 3" "REM 3")

* Kiểm định tổ hợp hệ số (lincom)
* (a) Mô hình 3: độ dốc của TPR ở nhóm DLI = 1 là α1 + α4
foreach m in POLS3 FEM3 REM3 {
    estimates restore `m'
    display as result "`m': độ dốc TPR khi DLI = 1"
    lincom TPRc + 1.DLI#c.TPRc
}
* (b) Mô hình 2: tác động biên của TPR tại LIQ p25, p50, p75 là α1 + α4 × (LIQ − trung bình)
quietly summarize LIQ, meanonly
local mL = r(mean)
_pctile LIQ, p(25 50 75)
local c1 = r(r1) - `mL'
local c2 = r(r2) - `mL'
local c3 = r(r3) - `mL'
foreach m in POLS2 FEM2 REM2 {
    estimates restore `m'
    foreach k in 1 2 3 {
        display as result "`m': tác động biên của TPR tại mức LIQ thứ `k'"
        lincom TPRc + `c`k''*c.TPRc#c.LIQc
    }
}

*==== 15. Kiểm định chọn mô hình: F, Breusch–Pagan LM, Hausman, Mundlak
local X1 "TPRc LIQc $control"
local X2 "TPRc LIQc $control TLI"
local X3 "TPRc DLI $control TLD"
forvalues m = 1/3 {
    display as result _n "===== Mô hình `m' ====="
    * (a) F test Pooled vs FEM: H0 tất cả u_i = 0
    quietly xtreg ATO `X`m'' i.year, fe
    display "F test u_i = 0: F(" e(df_a) ", " e(df_r) ") = " %9.3f e(F_f) "   p = " %9.4g Ftail(e(df_a), e(df_r), e(F_f))
    estimates store fe`m'
    * (b) Breusch–Pagan LM Pooled vs REM: H0 Var(u_i) = 0
    quietly xtreg ATO `X`m'' i.year, re
    estimates store re`m'
    xttest0
    * (c) Hausman FEM vs REM: H0 u_i không tương quan với X (REM phù hợp)
    hausman fe`m' re`m', sigmamore
    * (d) Mundlak – bản vững với gom cụm: thêm trung bình theo công ty của các biến X vào REM
    capture drop mn_*
    foreach v of local X`m' {
        bysort firm: egen mn_`v' = mean(`v')
    }
    quietly xtreg ATO `X`m'' mn_* i.year, re vce(cluster firm)
    testparm mn_*
}

*==== 16. Bản chất FEM: within transformation, LSDV và biến bị loại
* (a) Within bằng tay: trừ trung bình theo công ty rồi chạy OLS không hằng số
foreach v in ATO TPRc LIQc SIZ LEV CSH {
    capture drop w_`v' bar_`v'
    bysort firm: egen bar_`v' = mean(`v')
    gen w_`v' = `v' - bar_`v'
}
quietly xtreg ATO TPRc LIQc $control, fe
estimates store fe_khongnam
regress w_ATO w_TPRc w_LIQc w_SIZ w_LEV w_CSH, noconstant
* So sánh: hệ số giống hệt xtreg, fe (sai số chuẩn khác vì OLS tay không trừ bậc tự do của u_i)
estimates replay fe_khongnam
* (b) LSDV: OLS với biến giả từng công ty cho cùng hệ số; areg hấp thụ biến giả nên không vượt giới hạn số biến
areg ATO TPRc LIQc $control, absorb(firm)
* (c) Biến không đổi theo thời gian bị loại khỏi FEM
encode icb2, gen(nganh)
xtreg ATO TPRc LIQc $control i.year i.nganh, fe
* Stata báo các biến ngành "omitted because of collinearity": sau within chúng đều bằng 0.
* (d) Xem minh họa trên một doanh nghiệp
list ticker year ATO bar_ATO w_ATO TPRc bar_TPRc w_TPRc if ticker == "VNM", clean

*==== 17. Kiểm tra độ bền: biến giải thích trễ 1 năm
* L. lấy giá trị năm t − 1 của cùng công ty; nếu năm trước bị thiếu thì để trống.
gen L_TPR = L.TPR
gen L_LIQ = L.LIQ
gen L_SIZ = L.SIZ
gen L_LEV = L.LEV
gen L_CSH = L.CSH
foreach v in TPR LIQ {
    quietly summarize L_`v' if !missing(L_TPR, L_LIQ, L_SIZ, L_LEV, L_CSH), meanonly
    gen L_`v'c = L_`v' - r(mean) if !missing(L_TPR, L_LIQ, L_SIZ, L_LEV, L_CSH)
}
eststo LAG2: regress ATO c.L_TPRc##c.L_LIQc L_SIZ L_LEV L_CSH i.year, vce(cluster firm)
* Bảng so sánh pooled, FE và biến trễ
esttab M1 FE1 M2 FE2 M3 FE3 LAG2, b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) drop(*.year _cons, relax) r2 nogaps ///
    mtitles("M1 Pooled" "M1 FE" "M2 Pooled" "M2 FE" "M3 Pooled" "M3 FE" "M2 trễ 1 năm")
esttab M1 FE1 M2 FE2 M3 FE3 LAG2 using "so_sanh_pooled_FE.rtf", replace b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) drop(*.year _cons, relax) r2

* Biến phụ thuộc thay thế: ROA = lợi nhuận sau thuế / tổng tài sản
eststo ROAP1: regress ROA TPRc LIQc $control i.year, vce(cluster firm)
eststo ROAF1: xtreg ROA TPRc LIQc $control i.year, fe vce(cluster firm)
eststo ROAF2: xtreg ROA c.TPRc##c.LIQc $control i.year, fe vce(cluster firm)
eststo ROAF3: xtreg ROA c.TPRc##i.DLI $control i.year, fe vce(cluster firm)
* Thêm tăng trưởng doanh thu GRW (mất năm 2021). Cột SUB chạy cùng mẫu nhưng chưa có GRW để tách tác động của mẫu và của GRW.
eststo SUB1:  xtreg ATO TPRc LIQc $control i.year if !missing(GRW), fe vce(cluster firm)
eststo GRWF1: xtreg ATO TPRc LIQc $control GRW i.year, fe vce(cluster firm)
eststo GRWF2: xtreg ATO c.TPRc##c.LIQc $control GRW i.year, fe vce(cluster firm)
eststo GRWF3: xtreg ATO c.TPRc##i.DLI $control GRW i.year, fe vce(cluster firm)
esttab ROAP1 ROAF1 ROAF2 ROAF3 SUB1 GRWF1 GRWF2 GRWF3, b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) drop(*.year _cons, relax) r2 nogaps ///
    mtitles("ROA Pooled 1" "ROA FEM 1" "ROA FEM 2" "ROA FEM 3" "2022-25 FEM 1" "GRW FEM 1" "GRW FEM 2" "GRW FEM 3")
esttab ROAP1 ROAF1 ROAF2 ROAF3 SUB1 GRWF1 GRWF2 GRWF3 using "kiem_tra_do_ben_mo_rong.rtf", replace b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) drop(*.year _cons, relax) r2

*==== 18. Kiểm tra các giả định A1–A5
* A1 – tuyến tính: thêm TPRc bình phương vào mô hình 1
gen TPRc2 = TPRc^2
regress ATO TPRc LIQc $control TPRc2 i.year, vce(cluster firm)
* A2 – đa cộng tuyến: xem VIF ở bước 10 và ma trận tương quan ở bước 9 (|Corr| >= 0,8 hoặc VIF >= 10)
* A4 – phương sai sai số: Breusch–Pagan dạng n·R² cho mô hình 2 (H0: phương sai không đổi)
quietly regress ATO TPRc LIQc $control TLI i.year
estat hettest, rhs iid
* Kiểm định F (H0: R² = 0): hồi quy thường, không gom cụm, để có thống kê F cổ điển
regress ATO TPRc LIQc $control TLI i.year
* A3, A5: so sánh hệ số TPR giữa M2, FE2 và LAG2 ở bước 17; chọn mô hình ở bước 15

*==== 19. Lưu dữ liệu sau lọc
export delimited ticker year icb2 ATO TPR LIQ SIZ LEV CSH ROA GRW DLI using "du_lieu_sau_loc_stata.csv", replace
save "du_lieu_sau_loc.dta", replace

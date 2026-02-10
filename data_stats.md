# Data Statistics Log

### Preprocessing Run: 2026-02-10 16:38:40
**Raw Merged Data**
- Shape: (864304, 33)
- Price Mean: $267,905.68
- Price Median: $84,000.00

**Processed Data (Leakage Removed)**
- Shape: (864304, 25)
- Price Mean: $267,905.68
- Price Median: $84,000.00


### Feature Engineering Run: 2026-02-10 16:42:04
**Input Data**
- Shape: (864304, 25)
- Columns: PARID, TAXYR, SALEDT, BOOK, PAGE, INSTRUNO, INSTRTYP, INSTRTYP_DESC, PRICE, SALETYPE, STEB, STEB_DESC, APRTOT_x, EXTWALL_DESC, YRBLT, RMBED, FIXBATH, ROOF_COVER_DESC, SFLA, TOTAL_AREA, STORIES, LUC, LUC_DESC, NBHD, APRTOT_y

**Engineered Data**
- Shape: (864304, 34)
- Columns: PARID, TAXYR, SALEDT, BOOK, PAGE, INSTRUNO, INSTRTYP, INSTRTYP_DESC, PRICE, SALETYPE, STEB, STEB_DESC, APRTOT_x, EXTWALL_DESC, YRBLT, RMBED, FIXBATH, ROOF_COVER_DESC, SFLA, TOTAL_AREA, STORIES, LUC, LUC_DESC, NBHD, APRTOT_y, SaleYear, Month, HouseAge, HouseAge_Squared, SFLA_Squared, Efficiency_Ratio, Bed_Bath_Ratio, NBHD_Median_Size, Size_vs_NBHD


### Preprocessing Run: 2026-02-10 16:44:32
**Raw Merged Data**
- Shape: (864304, 33)
- Price Mean: $267,905.68
- Price Median: $84,000.00

**Processed Data (Leakage Removed)**
- Shape: (864304, 23)
- Price Mean: $267,905.68
- Price Median: $84,000.00


### Feature Engineering Run: 2026-02-10 16:47:30
**Input Data**
- Shape: (864304, 23)
- Columns: PARID, TAXYR, SALEDT, BOOK, PAGE, INSTRUNO, INSTRTYP, INSTRTYP_DESC, PRICE, SALETYPE, STEB, STEB_DESC, EXTWALL_DESC, YRBLT, RMBED, FIXBATH, ROOF_COVER_DESC, SFLA, TOTAL_AREA, STORIES, LUC, LUC_DESC, NBHD

**Engineered Data**
- Shape: (864304, 32)
- Columns: PARID, TAXYR, SALEDT, BOOK, PAGE, INSTRUNO, INSTRTYP, INSTRTYP_DESC, PRICE, SALETYPE, STEB, STEB_DESC, EXTWALL_DESC, YRBLT, RMBED, FIXBATH, ROOF_COVER_DESC, SFLA, TOTAL_AREA, STORIES, LUC, LUC_DESC, NBHD, SaleYear, Month, HouseAge, HouseAge_Squared, SFLA_Squared, Efficiency_Ratio, Bed_Bath_Ratio, NBHD_Median_Size, Size_vs_NBHD


### Preprocessing Run: 2026-02-10 17:06:01
**Raw Merged Data**
- Shape: (864304, 33)
- Price Mean: $267,905.68
- Price Median: $84,000.00

**Processed Data (Leakage Removed)**
- Shape: (864304, 23)
- Price Mean: $267,905.68
- Price Median: $84,000.00


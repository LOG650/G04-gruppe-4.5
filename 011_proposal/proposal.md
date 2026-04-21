# Project Proposal – LOG650 Group 4.5

## Group Members
- Gustavo Alfonso Holmedal  
- Thuy Thu Thi Tran  
- Inger Irgesund  

---

## Area
Retail business – Shoe store (real company).  
We use real data provided directly by the store owners.  
No hypothetical data will be used.

---

## Problem Statement
Inventory management – optimize purchasing decisions and stock levels  
to avoid using external storage for seasonal products.

We will evaluate which forecasting model is most suitable:
- Exponential Smoothing (ETS)
- ARIMA
- SARIMA

---

## Data
All relevant data is available:
- Inventory data  
- Purchasing data  
- Sales data  

---

## Decision Variables
- How many shoe pairs should be ordered?
- Is warehouse rebuilding necessary?

Primary focus: tactical purchasing decisions.  
We will later evaluate whether strategic decisions (warehouse redesign, shelf systems) are required.

---

## Objective Function
Minimize total costs and improve warehouse capacity.

Note: Full measurable results will not be available before 1.5 years, 
since autumn/winter 2026/2027 orders are already placed.

---

## Limitations
The project is limited to:
- Purchase quantity decisions
- Possible warehouse rebuilding
- Shoe product category only

The model includes:
- Purchasing costs
- Holding costs
- Rebuilding costs

The model does NOT include:
- Seasonal variation
- Delivery delays
- Price changes
- Shrinkage
- Campaigns
- Customer behavior

The model assumes known demand 
and focuses on a defined planning period.

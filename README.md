**Code For AVA Challenge@ICME 2023**
**Team: ALLAccept**

**Leaderboard: 0.61428 (top-3)**

See technique report in report.pdf (For more details please refer to thesis.pdf)

To reproduce the best result in competition, you could first download pre-trained models from:
```python
https://drive.google.com/drive/folders/1LTRXD44JZ6Y6t33EtnpkFYy0BGyr6WLl?usp=sharing
```
NOTE: We used two different sets of parameters for freeway and road task.
Once load them you can make predictions using:
```python
model = torch.load(p.model_file)['model']
all_pred, _, _, _, all_mil_pred = test_all_vis(testdata_loader, model, vis=True, device=device)
threhold = 0.078450665 if p.data_class == freeway else 0.42367747
pred = [max(pred) for pred in all_pred] if p.data_class == freeway else all_mil_pred
predcitions = pred > threhold
```


------------------------------------------------------------
To reproduce the experiment result in report: 

1. Extract Feature for model.
```python
bash ./feature-extract./run-extract_feature_new.py (remember to change your own dir path)
```
2. Run and save models.
```python
bash run-save.sh (remember to change your own dir path)
```
3. Prediction.
```python
bash run-ex.sh
```

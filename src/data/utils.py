from sklearn.metrics import roc_curve, precision_score, roc_auc_score, recall_score, auc

def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

def model_eval(predictions, ground_truth):
	roc = roc_curve(ground_truth, predictions)
	prec = precision_score(ground_truth, predictions)
	#roc_auc = roc_auc_score(ground_truth, predictions)
	recall = recall_score(ground_truth, predictions)
	#auc_score = auc(ground_truth, predictions)

	'''
	print(
        "########### Precision : {}, Recall: {}, ROC: {}, AUROC: {}, AUPRC: {} ###########".format(
            prec,
            recall,
            roc,
            roc_auc,
            auc_score
        )
    )
    '''
	print(
        "########### Precision : {}, Recall: {}, ROC: {} ###########".format(
            prec,
            recall,
            roc,
        )
    )

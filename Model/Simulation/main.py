
def msg(data):
    while True:
        OFS_choose = input(
            "Please choose an Online Feature Selection (OFS) algorithm: \n for FIRES press 1 \n for ABFS press 2 \n for MC-NN press 3")
        if OFS_choose == '1':
            try:
                tgt_index = int(input("Please enter your index target label (the classifier)"))
                fraction = int(input("Please enter a fraction value"))
                apply_fires(data,tgt_index,fraction)
            except Exception as e:
                print(e)

        elif OFS_choose == '2':  # ABFS
            pass

        elif OFS_choose == '3':  # MC-NN
            pass

        else:
            sys.stdout.write("invalid input, please try agian")

        OL_choose = input(
            "Please choose an Online Learning (OL) algorithm: \n for KNNClassifier press 1 \n for PerceptronMask press 2 \n for NaiveBayes press 3 \n for AdaptiveRandomForestClassifier press 4")
        try:
            if OL_choose == '1':
                n_neighbors = int(input("Please enter number of neighbors value"))
                max_window_size = int(input("Please enter max window size value"))
                leaf_size = int(input("Please enter leaf size value"))
                metric = "euclidean"

            elif OL_choose == '2':
                alpha = int(input("Please enter alpha value"))
                max_iter = int(input("Please enter max iteration value"))

            elif OL_choose == '3':  #TODO what are naive baise params?
                pass

            elif OL_choose == '4':
                n_estimators = int(input("Please enter number of estimators value"))
                lambda_value = int(input("Please enter lambda value"))
                split_confidence = int(input("Please enter split confidence value"))
                tie_threshold = int(input("Please enter tie_threshold value"))
                performance_metric = "acc"
                split_criterion = "info_gain"

            else:
                sys.stdout.write("invalid input, please try agian")
        except Exception as e:
            print(e)

if __name__ == "__main__":
    # apply_fires(df_name='file_path', tgt_index=0, epochs=1)
    while True:
        data_input = input("Please insert csv url path")
        try:
            data = pd.read_csv(data_input)
        except Exception as e:
            print(e)
        choice = input(
            "Please press 1 if you want stream. \n press 2 if you want batch.")

        if choice == '1':  # stream
            feature_choice = input(
                "Please press 1 if you want to run regular datastream \n press 2 if you want to use varying feature space \n press 3 for trapizodal.")
            chunk_size = input("enter chunk size")
            if feature_choice == '1':
                msg(data)
                pass

            elif feature_choice == '2':
                feature_precent = int(input("Choose feature % to randomize every time Chunk size"))
            elif feature_choice == '3':
                feature_precent = int(input("Choose feature % to add every time Chunk size"))

            else:
                sys.stdout.write("invalid input, please try again")
                continue

        elif choice == '2':   # shuffle from data, working on all features
            batch_size = int(input("enter batch size"))
            msg(data)

        else:  # bad key press
            continue

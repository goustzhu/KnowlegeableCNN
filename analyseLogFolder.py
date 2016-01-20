import os
import codecs
import string

def analyseLogFolder(path, label="1"):
    print "Started!"
    base_dir = path + "/" + label
    dir_list = os.listdir(base_dir)
    
    cell_word_statistic = dict()
    cell_scores = dict()
    
    dir_count = 0
    for d in dir_list:
        nn_dir = base_dir + "/" + d + "/nc_word/"
        print "\nFind ", nn_dir
        nn_num = 0
        while True:
            nn_file = nn_dir + str(nn_num) + "_merged"
#             print "Analyzing ", nn_file
            print ".",
            if os.path.exists(nn_file):
                with codecs.open(nn_file, "r", "utf-8", "ignore") as f:
                    cell_score = string.atof(f.readline().strip().split(" ")[1])
                    if not nn_num in cell_scores.keys():
                        cell_scores[nn_num] = cell_score
                    else:
                        cell_scores[nn_num] += cell_score
                    
                    f.readline()
                    # deal with nn-num cell
                    for line in f:
                        word, score = line.strip().split("\t")
                        score = string.atof(score)
                        
                        if not nn_num in cell_word_statistic.keys():
                            cell_state = dict()
                            cell_word_statistic[nn_num] = cell_state
                        else:
                            cell_state = cell_word_statistic[nn_num]
                        
                        if word in cell_state.keys(): 
                            word_state = cell_state[word]
                            word_state[0] += score
                            word_state[1] += 1 
                        else:
                            cell_state[word] = [score, 1]
            else:
                break
            nn_num += 1
        dir_count += 1
    print "Wirting result."
    analyse_base_dir = path + "/analyse_" + label + "/"     
    if not os.path.exists(analyse_base_dir):
        os.makedirs(analyse_base_dir)
    
    with codecs.open(analyse_base_dir + "cell_scores", "w", "utf-8", "ignore") as f:
        cell_scores_list = map(lambda (k, v): (k, v / dir_count) , list(cell_scores.items()))
        cell_scores_list.sort(key=lambda x:x[1], reverse=True)
        for c, score in cell_scores_list:
            f.write("%d\t%f\n" % (c, score))
            
    for (cell, cell_state) in cell_word_statistic.items():
        print "Wring cell: ", str(cell)
        with codecs.open(analyse_base_dir + str(cell), "w", "utf-8", "ignore") as f:
            word_list = map(lambda (k, v): (k, v[0] / v[1]) , list(cell_state.items()))
            word_list.sort(key=lambda x:x[1], reverse=True)
            for word, score in word_list:
                f.write("%s\t%f\n" % (word, score))
    

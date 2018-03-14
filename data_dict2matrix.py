import copy
import argparse
import time
from procces_data import prep_data


class make_train_data:
    def __init__(self, prep_ ):
        self.prep_ = prep_

    def dict2matrix(self, dict_filename = 'dict_data.csv',
                    matrix_filename = os.path.join('data', 'matrix_data.tsv')):
        print(dict_filename)
        start_time = time.time()
        f_r = open(dict_filename, 'r')
        f_w = open(matrix_filename, 'w')  # yes not properly with with

        info = []
        current_info = self.prep_.current_info
        current_info.append("_intent")
        t = "_ind\t_conv\t" + "\t".join(current_info) + '\n'
        print(t)
        f_w.write(t)
        intent_dict = {intent: str(i) for i, intent in enumerate(self.prep_.goal_legend)}

        for i, line in enumerate(f_r):
            if "\"%" in line:
                line = "1," + line
            line = line.replace("%", ",").replace("sql", "nlp").replace("\"", "")
            line = line.replace("\n", "").split(",")
            c_info = str(i) + "\t" + line[1]
            catagorien = {abc: "0" for abc in current_info}
            for j in range(2, len(line)):
                cat = line[j].split(":")[0]
                if cat in current_info:
                    value = line[j].split(":")[1]
                    if cat == "_intent":
                        value = value.replace(" ", "")
                        catagorien[cat] = intent_dict[value]
                    else:
                        catagorien[cat] = value
                    n_found = False

            if n_found:
                print("NOT found :::", line)

            for cat in current_info:
                c_info +=  "\t" + catagorien[cat]

    #         print("a",c_info)
            f_w.write(c_info + '\n')

        print(info)
        f_w.close()
        print("that took:", time.time() - start_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_data", type=str, default="", help="The original datafile from roger made with inky")
    args = parser.parse_args()

    prep_ = prep_data(dialog_class = None)
    make = make_train_data(prep_ = prep_)
    make.dict2matrix(dict_filename = args.dict_data)

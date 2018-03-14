import os
# import random
import pandas as pd
from chatbot import dialog
import argparse
# import copy
import numpy as np
import time
import sys
sys.path.insert(0,os.path.join(os.getcwd(), "va", "swagger_server"))

def file_len(fname):
    with open(fname, 'rb') as f:
        i = 0
        for a in f:
            i += 1
    return i + 1


class prep_data:
    def __init__(self, dialog_class, data_filename = None, 
                 batch_size = 32, with_extra_info = True,
                 chance_on_name = 0.90, threshold = 0.8):
    
        self.threshold = threshold
        print("procces_data with respect for timestep as word")
        self.with_extra_info = with_extra_info
        self.chance_on_name = with_extra_info
        self.batch_size = batch_size
        if not data_filename is None:
            self.data_filename = data_filename
            if batch_size > 0:
                self.make_reader()
        else:
            if self.with_extra_info:
                self.extra_info_def()
        self.dialog_ = dialog_class
        self.first_time = True
        self.seed = 0

        self.goal_legend = ["intent_unkown", "intent_weather", "intent_sql_lookup",
                            "intent_sql_change", "intent_sql_count"]
        self.goal_legend += ["intent_dwg_" + str(i) for i in range(33)]
        self.goal_dict = {abc: i for i, abc in enumerate(self.goal_legend)}
        self.current_info = ['nlp_f_name', 'nlp_l_name', "nlp_loc", "nlp_time",
                             'nlp_street', 'nlp_postcode', "nlp_city",
                             'nlp_dob'] # and '_intent' but that is to give the intent
        self.make_lower_prob = 0.5
        print(self.goal_dict)
        # print(self.list_w_names)

    def extra_info_def(self):
        self.list_w_specific_words = [['weer', 'temperatuur', 'regen'], ['naam', "heet", 'mr', 'mvr']]
        self.list_w_citys = ["Aa en Hunze", "Aalburg", "Aalsmeer", "Aalten", "Achtkarspelen", "Alblasserdam", "Albrandswaard", "Alkmaar", "Almelo", "Almere", "Alphen aan den Rijn", "Alphen-Chaam", "Ameland", "Amersfoort", "Amstelveen", "Amsterdam", "Apeldoorn", "Appingedam", "Arnhem", "Assen", "Asten", "Baarle-Nassau", "Baarn", "Barendrecht", "Barneveld", "Bedum", "Beek", "Beemster", "Beesel", "Berg en Dal", "Bergeijk", "Bergen", "Bergen", "Bergen op Zoom", "Berkelland", "Bernheze", "Best", "Beuningen", "Beverwijk", "Binnenmaas", "Bladel", "Blaricum", "Bloemendaal", "Bodegraven-Reeuwijk", "Boekel", "Borger-Odoorn", "Borne", "Borsele", "Boxmeer", "Boxtel", "Breda", "Brielle", "Bronckhorst", "Brummen", "Brunssum", "Bunnik", "Bunschoten", "Buren", "Capelle aan den IJssel", "Castricum", "Coevorden", "Cranendonck", "Cromstrijen", "Cuijk", "Culemborg", "Dalfsen", "Dantumadiel", "De Bilt", "De Fryske Marren", "De Marne", "De Ronde Venen", "De Wolden", "Delft", "Delfzijl", "Den Haag", "Den Helder", "Deurne", "Deventer", "Diemen", "Dinkelland", "Doesburg", "Doetinchem", "Dongen", "Dongeradeel", "Dordrecht", "Drechterland", "Drimmelen", "Dronten", "Druten", "Duiven", "Echt-Susteren", "Edam-Volendam", "Ede", "Eemnes", "Eemsmond", "Eersel", "Eijsden-Margraten", "Eindhoven", "Elburg", "Emmen", "Enkhuizen", "Enschede", "Epe", "Ermelo", "Etten-Leur", "Ferwerderadiel", "Geertruidenberg", "Geldermalsen", "Geldrop-Mierlo", "Gemert-Bakel", "Gennep", "Giessenlanden", "Gilze en Rijen", "Goeree-Overflakkee", "Goes", "Goirle", "Gooise Meren", "Gorinchem", "Gouda", "Grave", "Groningen", "Grootegast", "Gulpen-Wittem", "Haaksbergen", "Haaren", "Haarlem", "Haarlemmerliede en Spaarnwoude", "Haarlemmermeer", "Halderberge", "Hardenberg", "Harderwijk", "Hardinxveld-Giessendam", "Haren", "Harlingen", "Hattem", "Heemskerk", "Heemstede", "Heerde", "Heerenveen", "Heerhugowaard", "Heerlen", "Heeze-Leende", "Heiloo", "Hellendoorn", "Hellevoetsluis", "Helmond", "Hendrik-Ido-Ambacht", "Hengelo", "Heumen", "Heusden", "Hillegom", "Hilvarenbeek", "Hilversum", "Hof van Twente", "Hollands Kroon", "Hoogeveen", "Hoorn", "Horst aan de Maas", "Houten", "Huizen", "Hulst", "IJsselstein", "Kaag en Braassem", "Kampen", "Kapelle", "Katwijk", "Kerkrade", "Koggenland", "Kollumerland en Nieuwkruisland", "Korendijk", "Krimpen aan den IJssel", "Krimpenerwaard", "Laarbeek", "Landerd", "Landgraaf", "Landsmeer", "Langedijk", "Lansingerland", "Laren", "Leek", "Leerdam", "Leeuwarden", "Leiden", "Leiderdorp", "Leidschendam-Voorburg", "Lelystad", "Leudal", "Leusden", "Lingewaal", "Lingewaard", "Lisse", "Lochem", "Loon op Zand", "Lopik", "Loppersum", "Losser", "Maasdriel", "Maasgouw", "Maassluis", "Maastricht", "Marum", "Medemblik", "Meerssen", "Meierijstad", "Meppel", "Middelburg", "Midden-Delfland", "Midden-Drenthe", "Midden-Groningen", "Mill en Sint Hubert", "Moerdijk", "Molenwaard", "Montferland", "Montfoort", "Mook en Middelaar", "Neder-Betuwe", "Nederweert", "Neerijnen", "Nieuwegein", "Nieuwkoop", "Nijkerk", "Nijmegen", "Nissewaard", "Noord-Beveland", "Noordenveld", "Noordoostpolder", "Noordwijk", "Noordwijkerhout", "Nuenen, Gerwen en Nederwetten", "Nunspeet", "Nuth", "Oegstgeest", "Oirschot", "Oisterwijk", "Oldambt", "Oldebroek", "Oldenzaal", "Olst-Wijhe", "Ommen", "Onderbanken", "Oost Gelre", "Oosterhout", "Ooststellingwerf", "Oostzaan", "Opmeer", "Opsterland", "Oss", "Oud-Beijerland", "Oude IJsselstreek", "Ouder-Amstel", "Oudewater", "Overbetuwe", "Papendrecht", "Peel en Maas", "Pekela", "Pijnacker-Nootdorp", "Purmerend", "Putten", "Raalte", "Reimerswaal", "Renkum", "Renswoude", "Reusel-De Mierden", "Rheden", "Rhenen", "Ridderkerk", "Rijssen-Holten", "Rijswijk", "Roerdalen", "Roermond", "Roosendaal", "Rotterdam", "Rozendaal", "Rucphen", "Schagen", "Scherpenzeel", "Schiedam", "Schiermonnikoog", "Schinnen", "Schouwen-Duiveland", "'s-Hertogenbosch", "Simpelveld", "Sint Anthonis", "Sint-Michielsgestel", "Sittard-Geleen", "Sliedrecht", "Sluis", "Smallingerland", "Soest", "Someren", "Son en Breugel", "Stadskanaal", "Staphorst", "Stede Broec", "Steenbergen", "Steenwijkerland", "Stein", "Stichtse Vecht", "Strijen", "Sudwest-Fryslan", "Ten Boer", "Terneuzen", "Terschelling", "Texel", "Teylingen", "Tholen", "Tiel", "Tilburg", "Tubbergen", "Twenterand", "Tynaarlo", "Tytsjerksteradiel", "Uden", "Uitgeest", "Uithoorn", "Urk", "Utrecht", "Utrechtse Heuvelrug", "Vaals", "Valkenburg aan de Geul", "Valkenswaard", "Veendam", "Veenendaal", "Veere", "Veldhoven", "Velsen", "Venlo", "Venray", "Vianen", "Vlaardingen", "Vlieland", "Vlissingen", "Voerendaal", "Voorschoten", "Voorst", "Vught", "Waadhoeke", "Waalre", "Waalwijk", "Waddinxveen", "Wageningen", "Wassenaar", "Waterland", "Weert", "Weesp", "Werkendam", "West Maas en Waal", "Westerveld", "Westervoort", "Westerwolde", "Westland", "Weststellingwerf", "Westvoorne", "Wierden", "Wijchen", "Wijdemeren", "Wijk bij Duurstede", "Winsum", "Winterswijk", "Woensdrecht", "Woerden", "Wormerland", "Woudenberg", "Woudrichem", "Zaanstad", "Zaltbommel", "Zandvoort", "Zederik", "Zeewolde", "Zeist", "Zevenaar", "Zoetermeer", "Zoeterwoude", "Zuidhorn", "Zuidplas", "Zundert", "Zutphen", "Zwartewaterland", "Zwijndrecht", "Zwolle"]
        self.list_w_citys = [city.lower() for city in self.list_w_citys]
        start_time = time.time()
        self.list_w_f_names = []
        with open("list_w_names.txt", 'r') as f:
            for name in f:
                if self.batch_size == 0 or np.random.rand() > self.chance_on_name - 0.1:
                    self.list_w_f_names.append(name.replace("\n", "").lower())
        self.list_w_f_names.append("tako")
        self.list_w_f_names.append("roger")
        self.list_w_f_names.append("melis")
        self.list_w_f_names.append("ratko")
        print("the first name routine cost ",time.time() - start_time, " seconds")
        print(os.getcwd())
        self.list_w_l_names = []
        with open("list_w_last_names.txt", 'r') as f:
            for name in f:
                if self.batch_size == 0 or np.random.rand() >self.chance_on_name :
                    self.list_w_l_names.append(name.replace("\n", "").lower())
        self.list_w_l_names.append("tabak")
        self.list_w_l_names.append("daalen")
        self.list_w_l_names.append("schaap")
        self.list_w_l_names.append("popovski")
        print("the last name routine cost ",time.time() - start_time, " seconds")

    def make_reader(self, sep = "\t"):
        self.reader = pd.read_csv(self.data_filename, chunksize = self.batch_size, sep=sep)
        if self.with_extra_info:
            self.extra_info_def()

    def make_datapoint(self, df_dict, direct = True):
        # print(df_dict)
        #dialog in chat
        if np.random.rand() < self.make_lower_prob and not direct:
            for key in df_dict:
                df_dict[key] = df_dict[key].lower()

        #the output is going directly into the nn meaning we want it to have 3 dimensions
        # and we want an sentence_dict to make retriving words much faster
        array_, add_vector, sequence_length_, sentence_dict = self.dialog_.sentence2int(df_dict["_conv"], sentence_dict = True)
        print(sequence_length_)
        if self.with_extra_info:
            add_vector = self.check_if_in_matchlists(sentence_dict, add_vector)
        #combine the array_ +  add_vector
        array, sequence_length_ = self.dialog_._to_max_words(array_, add_vector, sequence_length_)

        list_w_keys = []
        for key in list(df_dict.keys()):
            if key in self.current_info:
                list_w_keys.append(key)

        if len(list_w_keys) == 0:
            return np.array([int(sequence_length_)])[np.newaxis, :], array[np.newaxis, :, :], sentence_dict
        # print(list_w_keys)
        y = np.zeros((self.dialog_.max_words))
        for j, key in enumerate(list_w_keys):

            if not df_dict[key] == "0" and not df_dict[key] == 0:
                # for _word in df_dict[key].split(" "):
                _word = str(df_dict[key])
                array_name, add_vector_name, sequence_length_name = self.dialog_.sentence2int(_word)
                if add_vector_name[-1, 39] == 1:
                    # <SPACE>
                    array_name, add_vector_name, sequence_length_name = array_name[:-1, :], add_vector_name[:-1, :], sequence_length_name - 1
                array_name, sequence_length_name = self.dialog_._to_max_words(array_name, add_vector_name, sequence_length_name)
                start_name, finish_name = self.find_start_and_finish(array[:sequence_length_,:], array_name[:sequence_length_name,:] )
                # print(key, j)
                if start_name == 0 and finish_name == 0:
                    print("conv_str", df_dict["_conv"])
                    print("name_str", df_dict[key])

                y = self.make_y(y, start_name, finish_name, j + 1)
        if '_intent' in df_dict:
            df_dict['_intent'] = self.goal_dict[df_dict['_intent']]
        else:
            df_dict['_intent'] = 0
            
        if direct:
            print(np.array(df_dict['_intent']).shape)
            intent = np.array(df_dict['_intent'])[np.newaxis,np.newaxis]
            return intent, y[np.newaxis,:], np.array([int(sequence_length_)])[np.newaxis,:], array[np.newaxis,:,:], sentence_dict
        else:
            return df_dict['_intent'][np.newaxis], y, np.array([int(sequence_length_)]), array

    def get_batch(self,seed = None, get_a_conv = False):
        if seed is None:
            self.seed += 1
            np.random.seed(self.seed)
        else:
            np.random.seed(seed)

        if self.batch_size == 0:
            df = pd.read_csv(self.data_filename).reset_index(drop = True)

            self.batch_size = df.shape[0]
        else:
            try:
                df = next(self.reader).reset_index(drop = True)
            except:
                self.make_reader()
                df = next(self.reader).reset_index(drop = True)

        if self.first_time:
            print(df.head())
            start_time = time.time()
        y = np.zeros((self.batch_size,self.dialog_.max_words))
        seq_len = np.zeros([self.batch_size,1])
        array = np.zeros([self.batch_size,self.dialog_.max_words,self.dialog_.input_depth])
        goal = np.zeros((self.batch_size,1))
        for i in range(df.shape[0]):

            g_, y_, sequence_length_, array_ = self.make_datapoint(df.iloc[i,:].to_dict(), direct = False)
            y[i,:] = y_
            seq_len[i,:] = sequence_length_
            array[i,:,:] = array_
            goal[i,:] = g_
        if self.first_time:
            print("this took: ", time.time() - start_time)
            self.first_time = False
        return_list = [goal, y, seq_len, array]
        if get_a_conv:
            return return_list + [df.loc[0, "_conv"]]
        return return_list
    def make_y(self, y, start_j, finish_j, j):
        y[int(start_j):int(finish_j)] = j
        return y
    def find_start_and_finish(self, array, name_array ):
        #array 100*340
        #name_array ?*340
        name_len = name_array.shape[0] #12
        i = array.shape[0]-name_len # 100 - 12 = 88
        while i >= 0:
            if np.array_equal(array[i:i+name_len,:], name_array): # 88: 100
                return i, i + name_len
            i -= 1

        # TypeError("failed to find array in conversation")
        print("FAILED to find name in converation")
        print("array :,1",array[:,301])
        print("name_array -1,1",name_array[:,301])
        print("reverse name:", self.dialog_.int2sentence(name_array, seq_len = name_array.shape[0]), "reverse conv:", self.dialog_.int2sentence(array, seq_len = array.shape[0]))
        return 0, 0
    def check_if_in_matchlists(self, sentence_dict, add_vector):
        word_out_of_letters = ""
        for i, word in enumerate(list(sentence_dict.values())):
            word = word.lower()
            if len(word) == 1 and not word == "<SPACE>":
                word_out_of_letters += word
                start_of_word = i
            elif word == "<SPACE>":
                word = word_out_of_letters
                word_out_of_letters = ""
                # print(word)
            if len(word) > 1:
                start_of_word = i
            if word in self.list_w_citys:
                add_vector[start_of_word:i,self.dialog_.alphabet["<city>"]] = 1
                # print("found '", word, "' out sentence_dict in list_w_citys")
            if word in self.list_w_f_names:
                add_vector[start_of_word:i,self.dialog_.alphabet["f_name"]] = 1
                # print("found '", word, "' out sentence_dict in list_w_f_names")
            if word in self.list_w_l_names:
                add_vector[start_of_word:i,self.dialog_.alphabet["l_name"]] = 1
                # print("found '", word, "' out sentence_dict in list_w_l_names")

        return add_vector    

    def get_info(self, y = None, x = None, seq_len= None, intent = [], sen_dict = None, plot = False):
        if len(y.shape) == 3:
            assert y.shape[0] == 1
            y = y[0,:,:]
            x = x[0,:,:]
        seq_len = np.squeeze(seq_len)
        y = y[:seq_len, :]
        x = x[:seq_len, :]
        y_ = np.zeros((y.shape[0], len(self.current_info)))
        print(y_.shape,y.shape) #  (7, 8) (7, 20)
        y_[:y.shape[0], :y.shape[1]] = y
        y = y_
        dict_w_names = {}
        for j in range(len(self.current_info)):
            confidence = np.max(y[:,j])
            if confidence > self.threshold:
                mask = np.where(y[:,j] > self.threshold)
                if sen_dict == None:
                    x_info = x[mask,:]
                    dict_w_names[self.current_info[j]] = {"text":self.dialog_.int2sentence(x_info, seq_len = x_info.shape[0]),
                                                          "confidence":confidence}
                else:
                    self.current_info[j]
                    dict_w_names[self.current_info[j]] = {"text":" ".join([sen_dict[interger] for interger in mask[0]]), 
                                                    "confidence":confidence}

        #goal
        if not isinstance(intent,list):
            # asumiong a shape of (1,num_intents)
            i = np.argmax(intent[0,:])
            confidence = intent[0,i]
            if confidence > self.threshold:
                dict_w_names[self.goal_legend[i]] = {"text": True,
                                          "confidence": confidence}
            else:
                dict_w_names["intent_unknown"] = {"text": True,
                                          "confidence": 0}
        if plot:
            self.plot_sentence_result(y, sen_dict, intent)
        return dict_w_names

    def plot_sentence_result(self, y_, sens_dict, g = None):
        # test
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,3))
        fig.suptitle('Sentence', fontsize=14)
        ax = fig.add_subplot(111)
        ax.axis([0, 10, 0, 1])
        inv = ax.transData.inverted()
        cur_word = ""
        prev_x1 = [np.array([0.3, 0.5])]
        for key,text_item in sens_dict.items():

            r = fig.canvas.get_renderer()
            t = ax.text(prev_x1[-1][0], prev_x1[-1][1], text_item + "  ")
            ax.text(prev_x1[-1][0], prev_x1[-1][1], text_item + "  ")
            bb = t.get_window_extent(renderer=r)

            prev_x1.append( inv.transform((bb.x1, bb.y0)))
            prev_x1[-1][1] = 0.5
        #         plt.bar()

        bar_matrix = np.array(prev_x1)
        ind = [(bar_matrix[i+1,0] +bar_matrix[i,0])/2 for i in range(bar_matrix.shape[0]-1)]
        width = [bar_matrix[i+1,0] -bar_matrix[i,0] for i in range(bar_matrix.shape[0]-1)]
        print(len(ind), y_[:,0].shape, len(width))
        a = [plt.bar(ind, y_[:,0], width)]
        for i in range(1,y_.shape[1]):

            a.append(plt.bar(ind, y_[:, i], width, bottom = np.sum(y_[:,:i-1],axis = -1)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.legend(self.current_info, loc = 8, bbox_to_anchor=(1.05, 0.2))
        plt.axis('off')
        plt.show()
        if g is not None:
            b = [plt.bar(x = 0.5, height = g[0, 0])]
            for i in range(1,g.shape[1]):
                b.append(plt.bar(x = 0.5,height = g[0,i],bottom = np.sum(g[0,:(i-1)])))
            plt.legend(self.goal_legend, loc = 8, bbox_to_anchor=(1.05, 0.2))
            plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--word2vec", type=int, default=1, help="loading word2vec")
    parser.add_argument("--data_filename", type=str, default="trainings_data_w_multi_class_f2.0_.tsv", help="data_filename")
    args = parser.parse_args()

    import time

    dialog_ = dialog(word2vec = args.word2vec == 1)
    prep = prep_data( batch_size = 64, dialog_class= dialog_) #data_filename=args.data_filename,

    sequence_length_, array, sentence_dict = prep.make_datapoint({"_conv":"hello hello Alphen aan den Rijn"})
    print(sequence_length_.shape, array.shape, sentence_dict)
    start_time = time.time()
    y = np.random.rand(1,100,20)-0.2
    y[0,2:,1] = 0.99
    result = prep.get_info(y = y, x=array, seq_len=sequence_length_, sen_dict = sentence_dict)
    print(result)
    print("this took:", time.time()-start_time)
    # train = prep_data(data_filename='trainings_data_w_multi_class_f2.0_.tsv', batch_size = 32, dialog_class= dialog_)
    # for _ in range(20):
    #     y, sequence_length_, array = train.get_batch(seed = 0)

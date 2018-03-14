import connexion
import six

from swagger_server.models.named_entities_response import NamedEntitiesResponse  # noqa: E501
from swagger_server.models.named_entities_response import NamedEntity
from swagger_server import util
from swagger_server.sess_ginfo import getSess
from cgitb import text

#------ code insert WS ------
import procces_data


def tfInference(text):
    extra_info_num_of_sparce_e = 10
    sess, ginfo = getSess()
    # print(ginfo)
 # feel free to adapt the conv enty and the first name (info_1) and last name (info_2) you want it to return
    # with making the datapoints I do not use the dict which start with _ for making identifing entities
    seq_len, x_, sens_dict =  ginfo['procces'].make_datapoint({"_conv": (text)})

    # run the model with the datapoint generated
    feed_dict = {ginfo["X"]: x_[:, :, :-extra_info_num_of_sparce_e],
                 ginfo["extra_info_place"]: x_[:, :, -extra_info_num_of_sparce_e:],
                 ginfo["s_sequence_length"]: seq_len,
                 ginfo["keep_prob"]: 1}

    # print(feed_dict)
    y_, g_ = sess.run([ginfo['logits'], ginfo['pred_goal']], feed_dict = feed_dict )
#     print(np.round(y_,1))

    return ginfo['out'].get_info(y = y_, x = x_, seq_len= seq_len, goal = g_, sen_dict = sens_dict, plot = False)
#     y_output_model = y_

#------ code insert WS ------


def get_entities(text):  # noqa: E501
    """Custom Named Entities WebAPI. Returns trained named entities.

    Obtain custom entities from text # noqa: E501

    :param text: Text to analyse for occurence of Custom Entities
    :type text: str

    :rtype: NamedEntitiesResponse
    """
    # a= b
    status = tfInference(text)
    list_with_entries = []
    for key in status:
        print(key, status[key])
        list_with_entries.append(NamedEntity(key, status[key]['text'], status[key]['confidence']))
    return NamedEntitiesResponse(list_with_entries, str(status))

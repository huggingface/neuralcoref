## Note for PII Hackathon

This is only a partial copy of the Masakhane-NER code as of Sept 26, 2021 for the PII Hackathon. This code is a reference implementation for Module 3. Only the code under this repo will be supported for the hackathon.

For reference, the complete Masakhane-NER code please refer to https://github.com/masakhane-io/masakhane-ner

Original README:

## [MasakhaNER: Named Entity Recognition for African Languages](https://arxiv.org/abs/2103.11811)

This repository contains the code for [training NER models](https://github.com/masakhane-io/masakhane-ner/tree/main/code), scripts to [analyze the NER model predictions](https://github.com/masakhane-io/masakhane-ner/tree/main/analysis_scripts) and the [NER datasets](https://github.com/masakhane-io/masakhane-ner/tree/main/data) for all the 10 languages listed below. 

The code is based on HuggingFace implementation.

### Required dependencies
* python
  * [transformers](https://pypi.org/project/transformers/) : state-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.
  * [seqeval](https://pypi.org/project/seqeval/) : testing framework for sequence labeling.
  * [ptvsd](https://pypi.org/project/ptvsd/) : remote debugging server for Python support in Visual Studio and Visual Studio Code.

```bash
pip install transformers seqeval ptvsd
```

### Volunteers
----------------
| Language | Volunteer names |
|----------|-----------------|
| Amharic | Seid Muhie Yimam, Musie Meressa, Israel Abebe, Degaga Wolde, Henok Tilaye, Dibora Haile  |
| Hausa  | Shamsudden Muhammad, Tajuddeen Rabiu Gwadabe, Emmanuel Anebi|
| Igbo  | Ignatius Ezeani, Chris Emezue, Chukwuneke Chiamaka, Nkiru Odu, Amaka, Isaac |
| Kinyarwanda | Rubungo Andre Niyongabo, Happy Buzaaba |
|Luganda   |  Joyce Nabende, Jonathan Mukiibi, Eric Peter Kigaye, Ivan Ssenkungu, Ibrahim Mbabaali, Batista Tobius, Maurice Katusiime, Deborah Nabagereka, Tobius Saolo |
| Luo   | Perez Ogayo, Verrah Otiende |
| Naija Pidgin | Orevaoghene Ahia, Kelechi Ogueji, Adewale	Akinfaderin, Aremu Adeola Jr., Iroro Orife, Temi Oloyede, Samuel Abiodun Oyerinde, Victor Akinode   |
| Swahili | Catherine Gitau, Verrah Otiende, Davis David, Clemencia Siro, Yvonne Wambui, Gerald Muriuki  |
| Wolof | [Abdoulaye Diallo](https://github.com/abdoulsn), [Thierno Ibrahim Diop](https://github.com/bayethiernodiop), and [Derguene Mbaye](https://github.com/DerXter), Samba Ngom, Mouhamadane Mboup  |
| Yorùbá | David Adelani, Mofetoluwa Adeyemi, Jesujoba Alabi, Tosin Adewumi, Ayodele Awokoya |

If you make use of this dataset, please cite us:

### BibTeX entry and citation info
```
@misc{adelani2021masakhaner,
      title={MasakhaNER: Named Entity Recognition for African Languages}, 
      author={David Ifeoluwa Adelani and Jade Abbott and Graham Neubig and Daniel D'souza and Julia Kreutzer and Constantine Lignos and Chester Palen-Michel and Happy Buzaaba and Shruti Rijhwani and Sebastian Ruder and Stephen Mayhew and Israel Abebe Azime and Shamsuddeen Muhammad and Chris Chinenye Emezue and Joyce Nakatumba-Nabende and Perez Ogayo and Anuoluwapo Aremu and Catherine Gitau and Derguene Mbaye and Jesujoba Alabi and Seid Muhie Yimam and Tajuddeen Gwadabe and Ignatius Ezeani and Rubungo Andre Niyongabo and Jonathan Mukiibi and Verrah Otiende and Iroro Orife and Davis David and Samba Ngom and Tosin Adewumi and Paul Rayson and Mofetoluwa Adeyemi and Gerald Muriuki and Emmanuel Anebi and Chiamaka Chukwuneke and Nkiruka Odu and Eric Peter Wairagala and Samuel Oyerinde and Clemencia Siro and Tobius Saul Bateesa and Temilola Oloyede and Yvonne Wambui and Victor Akinode and Deborah Nabagereka and Maurice Katusiime and Ayodele Awokoya and Mouhamadane MBOUP and Dibora Gebreyohannes and Henok Tilaye and Kelechi Nwaike and Degaga Wolde and Abdoulaye Faye and Blessing Sibanda and Orevaoghene Ahia and Bonaventure F. P. Dossou and Kelechi Ogueji and Thierno Ibrahima DIOP and Abdoulaye Diallo and Adewale Akinfaderin and Tendai Marengereke and Salomey Osei},
      year={2021},
      eprint={2103.11811},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

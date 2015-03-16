##When Crowdsourcing Fails

Code for the 2015 ASME Journal of Mechanical Design Paper - When Crowdsourcing Fails: A Study of Expertise on Crowdsourced Design Evaluation

[Link to the Paper](http://mechanicaldesign.asmedigitalcollection.asme.org/mobile/article.aspx?articleid=1935529)

<img src="https://www.aburnap.com/images/research_logos/crowd_evaluation.png" alt="crowd_expertise_plot" style="width:400px; text-align:center;">


This paper investigates crowdsourced evaluation for engineering designs using both simulated and real human crowds.  The crowd's combined evaluation is aggregated using either averaging or a Bayesian network crowd consensus model that accounts for evaluator expertise and design difficulty.

The key contribution of this work is showing an example of when crowdsourcing may not give a very accurate evaluation even on a "simple" engineering task.  This is shown to regardless of crowd consensus model used, due to experts in the crowd being washed out by "consistently wrong" non-experts.

## Code Example

To replicate paper experiments, choose either ./simulated_crowds or ./human_crowds and run the respective python run code from a terminal.

```python
# Example
python ./human_crowds/human_crowd_study.py
```

## Installation

This code depends on other python packages. To run this code, please install using the following:

```python
pip install -r requirements.txt

```

## License

The code is licensed under the MIT license. Feel free to use this code for your research.  If you find this code useful, please use the following citation information:

Burnap, A., Ren, Y., Gerth, R., Papazoglou, G., Gonzalez, R., & Papalambros, P. Y. (2015). When Crowdsourcing Fails: A Study of Expertise on Crowdsourced Design Evaluation. Journal of Mechanical Design, 137(3), 031101.

@article{burnap2015crowdsourcing,
  title={When Crowdsourcing Fails: A Study of Expertise on Crowdsourced Design Evaluation},
  author={Burnap, Alex and Ren, Yi and Gerth, Richard and Papazoglou, Giannis and Gonzalez, Richard and Papalambros, Panos Y},
  journal={Journal of Mechanical Design},
  volume={137},
  number={3},
  pages={031101},
  year={2015},
  publisher={American Society of Mechanical Engineers}
}




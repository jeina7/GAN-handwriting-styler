# My Handwriting Styler | 내 손글씨를 따라쓰는 인공지능

<figure>
  <img src="pngs/results_rigid_fonts.png" width="750">
  <figcaption style="text-align: center;">왼쪽 : 모델이 생성한 가짜 이미지 / 오른쪽 : 실제 폰트 이미지</figcaption>
</figure>

## \# 프로젝트 소개
GAN 기반으로 된 모델을 활용해 사람의 손글씨를 학습하고 그 글씨체를 반영한 글자 이미지를 생성합니다. 사람의 손글씨를 학습하기 전에, 먼저 대량의 컴퓨터 폰트 글자 이미지로 사전 학습을 진행하고, 그 후 적은 양의 사람 손글씨 데이터로 Transfer Learning(전이학습)을 진행합니다. 사전학습의 과정은 [kaonashi-tyc](https://github.com/kaonashi-tyc)가 중국어로 진행한  [zi2zi](https://github.com/kaonashi-tyc/zi2zi) 프로젝트의 도움을 받았습니다.  


프로젝트의 전체 진행 과정, 이론, 실험 등은 [블로그 글](https://jeinalog.tistory.com/15)에서 더 자세히 읽으실 수 있습니다. 

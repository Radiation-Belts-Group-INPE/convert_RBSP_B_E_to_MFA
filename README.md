# Conversão de Coordenadas para o Sistema MFA (Mean Field-Aligned)

Este repositório contém o código para processar dados de magnetômetro (EMFISIS) e campo elétrico (EFW) dos satélites Van Allen Probes (RBSP), realizando a filtragem de sinais e a rotação de coordenadas para o sistema **MFA**.

## 1. Por que usamos o sistema MFA?
No espaço geofísico, os dados brutos costumam vir em coordenadas cartesianas globais (como **GSM** ou **GSE**). No entanto, para estudar ondas ULF (Ultra-Low Frequency) e a dinâmica do plasma na magnetosfera, precisamos analisar as flutuações em relação ao **campo magnético local de fundo**. 

O sistema MFA separa fisicamente as flutuações em três componentes ortogonais:
* **Paralela ($\parallel$):** Componente compressional (na direção do campo médio).
* **Radial/Poloidal ($r$):** Componente perpendicular apontando para fora (em direção ao gradiente de L-shell).
* **Azimutal/Toroidal ($\phi$):** Componente perpendicular apontando para leste (direção azimutal).

---

## 2. Passo a Passo do Processamento

### Passo 1: Limpeza do Campo Elétrico ($\vec{V} \times \vec{B}$)
O movimento do satélite corta as linhas de campo magnético a alta velocidade, gerando um campo elétrico artificial induzido que mascara as ondas reais. A primeira coisa a fazer é remover esse efeito do campo elétrico medido ($\vec{E}_{medido}$):

$$\vec{E}_{corrigido} = \vec{E}_{medido} - (\vec{V}_{sc} \times \vec{B}_{total})$$

> **Nota:** É crucial usar o vetor velocidade do satélite ($\vec{V}_{sc}$) e o campo magnético total medido ($\vec{B}_{total}$) nesta etapa.

### Passo 2: Isolamento das Flutuações (Detrending)
Para estudar ondas (ex: Pc5, na banda de 2 a 7 mHz), precisamos remover a variação lenta da órbita do satélite (o "campo de fundo"). Aplicamos um filtro digital passa-alta (Butterworth) com frequência de corte típica em $10\text{ mHz}$.
* **Campo de Fundo ($\vec{B}_0$):** O que sobra abaixo da frequência de corte (tendência orbital).
* **Perturbações ($\delta\vec{B}$ e $\delta\vec{E}$):** O sinal oscilatório puro acima da frequência de corte.

### Passo 3: Construção do Triedro MFA (Geometria)
A matriz de rotação ponto a ponto é calculada usando exclusivamente o **campo magnético de fundo ($\vec{B}_0$)** e a **posição do satélite ($\vec{R}_{sat}$)** em coordenadas GSM:

1. **Vetor unitário paralelo ($\hat{e}_\parallel$):** Alinhado ao campo de fundo médio.
   $$\hat{e}_\parallel = \frac{\vec{B}_0}{|\vec{B}_0|}$$
2. **Vetor unitário azimutal ($\hat{e}_\phi$):** Perpendicular ao plano formado pelo campo magnético e a posição do satélite.
   $$\hat{e}_\phi = \frac{\hat{e}_\parallel \times \vec{R}_{sat}}{|\hat{e}_\parallel \times \vec{R}_{sat}|}$$
3. **Vetor unitário radial ($\hat{e}_r$):** Fecha o triedro ortogonal de mão direita.
   $$\hat{e}_r = \hat{e}_\phi \times \hat{e}_\parallel$$

### Passo 4: Projeção dos Vetores (Rotação)
Finalmente, montamos a matriz de rotação $R = [\hat{e}_\parallel, \hat{e}_r, \hat{e}_\phi]^T$ e projetamos as perturbações isoladas no passo 2 para obter os componentes físicos finais:

$$\vec{V}_{MFA} = R \cdot \vec{V}_{GSM}$$

Isso é aplicado tanto para as flutuações magnéticas ($\delta\vec{B}$) quanto elétricas ($\delta\vec{E}$).

---

## 3. Pré-requisitos para Rodar o Código
Certifique-se de que as bibliotecas científicas básicas e o `speasy` (para baixar os dados reais automaticamente do CDAWeb) estejam instalados:

```bash
pip install numpy scipy pandas matplotlib speasy
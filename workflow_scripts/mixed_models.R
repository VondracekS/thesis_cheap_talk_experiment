# Simulation evaluation

# libraries
library(lme4)
library(flexplot)
library(dplyr)
library(ggplot2)
library(dagitty)
library(ggdag)
library(ggplot2)
library(modelsummary)
library(lmtest)
library(xtable)
library(car)
library(lfe)


# load the experiment data
read.csv("./data/data_processed/gameplay_full_cmpliant.csv") %>% 
  select(-"X") -> df_simulation

# load info about participants
read.csv("data/data_processed/data_info.csv") %>% select(-"X") -> df_info

# join the two data frames
df_info %>% inner_join(df_simulation) -> df_simulation

# create the dags:
dag_payoff <- dagify(TARGET ~ treatment + cost + opp_cost + gender + 
                       econ + id + age + correct + pred + action + prediction,
                     action ~ id,
                     prediction ~ id,
                     correct ~ id,
                     exposure = "treatment",
                     outcome = "TARGET")

dag_payoff %>% 
  tidy_dagitty() %>% 
  ggplot(aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_dag_point(size=30, colour='DarkBlue') +
  geom_dag_edges() +
  geom_dag_text(col = "white", size = 2.9) +
  scale_color_grey() +
  theme_dag()


# SImulation models
# logit 
df_simulation %>% 
  glmer(data = ., formula = pareto ~ 1 + (1|id), family='binomial') -> baseline_pareto

df_simulation %>% 
  glmer(data = ., formula = pareto ~ male + age + econ_study + treatment + cost + opp +  correct + 
          p_choices + econ_study + (1|id), 
        family="binomial") -> logistic_pareto

df_simulation %>% 
  glmer(data = ., formula = pareto ~ male + age + econ_study + cost + opp +  correct + 
          p_choices + econ_study + (1|id), 
        family="binomial") -> logistic_pareto_wo_treatment#

# lrtest
lrtest(logistic_pareto, logistic_pareto_wo_treatment)



modelsummary(list("Baseline Model"=baseline_pareto,"Full Model"=logistic_pareto,
                  "Restrained Model"=logistic_pareto_wo_treatment),
             estimate = "{estimate}{stars}", output="latex")



# logit 
df_simulation %>% 
  glmer(data = ., formula = payoff ~ 1 + (1|id)) -> baseline_payoff

df_simulation %>% 
  lmer(data = ., formula = payoff ~ male + age + econ_study + treatment + cost + 
          opp +  correct + p_choices + econ_study + (1|id)) -> payoff

df_simulation %>% 
  lmer(data = ., formula = payoff ~ male + age + econ_study + cost + opp +  correct + 
          p_choices + econ_study + (1|id)) -> payoff_wo_treatment

# lrtest
xtable(tibble(lrtest(payoff, payoff_wo_treatment)))

modelsummary(list("Baseline Model"=baseline_payoff,"Full Model"=payoff,
                  "Restrained Model"=payoff_wo_treatment),
             estimate = "{estimate}{stars}", output="latex")

experiment_simulation %>% 
  group_by(treatment )%>% summarise(mean = mean(payoff)) %>% xtable(.)


# Actions analysis

# create the dags:
dag_payoff <- dagify(prob_B ~  cost + gender + econ + age + action_type,
                     exposure = "action_type",
                     outcome = "prob_B")

dag_payoff %>% 
  tidy_dagitty() %>% 
  ggplot(aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_dag_point(size=30, colour='DarkBlue') +
  geom_dag_edges() +
  geom_dag_text(col = "white", size = 2.9) +
  scale_color_grey() +
  theme_dag()



df_actions <- read.csv("data/data_processed/df_compliant_actions.csv")

df_actions <- df_actions %>% inner_join(df_info)

df_actions %>%
  filter(variable %in% c("p_choices_cm", "p_choices")) %>% 
  glm(data = .,
        formula = value ~ cost + treatment + male + age + econ_study,
        family='binomial') -> model_actions_cm_full

df_actions %>%
  filter(variable %in% c("p_choices_ca", "p_choices")) %>% 
  glm(data = .,
        formula = value ~ cost + treatment + male + age + econ_study,
        family='binomial') -> model_actions_ca_full

df_actions %>%
  filter(variable %in% c("prediction_cm", "prediction")) %>% 
  glm(data = .,
        formula = value ~ cost + treatment + male + age + econ_study,
        family='binomial') -> model_pred_cm_full

df_actions %>%
  filter(variable %in% c("prediction_ca", "prediction")) %>% 
  glm(data = .,
        formula = value ~ cost + treatment + male + age + econ_study,
        family='binomial') -> model_pred_ca_full

df_actions %>%
  filter(variable %in% c("p_choices_ca", "p_choices_cm", "p_choices")) %>% 
  glm(data = .,
        formula = value ~ cost + as.factor(variable) + male + age + econ_study,
        family='binomial') -> model_action_ca_cm

df_actions %>%
  filter(variable %in% c("prediction_ca", "prediction_cm", "prediction")) %>% 
  glm(data = .,
        formula = value ~ cost + as.factor(variable) + male + age + econ_study,
        family='binomial') -> model_pred_ca_cm


modelsummary(models=list("Action A"=model_actions_ca_full,
                         "Action B" = model_actions_cm_full,
                         "Action A vs Action B" = model_action_ca_cm),
                         estimate = "{estimate}{stars}",
             vcov = "HC")

modelsummary(models=list("Prediction A" = model_pred_ca_full,
                         "Prediction B" = model_pred_cm_full, 
                         "Prediction  A vs B" = model_pred_ca_cm),
             estimate = "{estimate}{stars}",
             vcov = "HC")




qqnorm(resid(model_pred_ca_full))



# Non lmer models

df_actions %>%
  filter(variable %in% c("p_choices_ca", "p_choices_cm", "p_choices")) %>% 
  glmer(data = .,
        formula = value ~ cost + as.factor(variable) + (1|id),
        family='binomial') -> model_actions_full



df_actions %>%
  filter(variable %in% c("p_choices_ca", "p_choices_cm", "p_choices")) %>% 
  glmer(data = .,
        formula = value ~ 1 + (1|id),
        family='binomial') -> model_actions_base



summary(model_actions)
linearHypothesis(model_actions, "as.factor(variable)p_choices_cm = as.factor(variable)p_choices_ca")


modelsummary(list("model_action"=model_actions_diff, "model_pred"=model_pred_diff), 
             estimate = "{estimate}{stars}")
modelsummary()


qqnorm(resid(model_pred_diff))

summary(df_actions)
  
  


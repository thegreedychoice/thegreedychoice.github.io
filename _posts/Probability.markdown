---
layout: post
title:  "Probability Theorygg"
date:   2017-08-06 01:05:31 -0400
categories: machine learning
permalink: /machinelearning/probability-theory-part1

---

<br/>

<h3 style="color:#2a7ae2"> Introduction </h3>
<section>
{% highlight ruby %}
Probability theory is nothing but common sense reduced to calculation.
-Pierre Laplace, 1812
{% endhighlight  %}

<br/><br/>
<p>
Probability theory is the core foundation of machine learning, and so its very important for you to familarize yourself with the key concepts. We all know heard phrases such as "the probability that a coin will flip head or tails is 0.5". What does it really mean? So basically, there are two schools of thought on the subject. One interpretation is the Frequentist View which measures probability of an event in terms of <strong>long run frequencies</strong> of that event. For instance, the above example could be interpreted as, if we flip a coin under the same circumstances for many times, the probability that it would be heads is half the number of times it is flipped (thus 0.5).
</p>

<p> 
&emsp;&emsp;The other interpretation is the Bayesian view which does not measures the probability of a certain event based on frequencies and experimentation but it attempts to <strong>quantify the uncertainity</strong> of that event. In that way, it is fundamentally more related to the informationa at hand rather than the repeated trials of a certain event. The coin example would be interpreted by a Bayesian as the probability of landing a heads is equal to the probability of landing a tails in the next toss. 
</p>

<p>
&emsp;&emsp;Both school of thoughts have been around for a long time in Statistics and usually they don't seem to like each other's view on the matter much but they both certainly have their advantages(Totally depends on the subject and constraints of the experiment). One big advantage of the Bayesian View could certainly be, that we can use it to model uncertainities about events that don't really happen a lot or have long run frequencies. For example, if we need to find the probability of polar ice cap melting by the year 2030, a frequentist won't be really of much help since the event hasn't even happened once, forget about its run frequency. But a bayesian's perspective can clearly come handy since we can use his approach and methods to help model the uncertainity of this event. The Bayesian view takes into consideration the whole randomness of this world we live in much better than a frequentist does who totally relies on some sort of pattern. Another good example is classification of spam messages, where the frequentist view would hardly be of any help as it relies on a lot of different paramters and not just the frequency. 
	But the good thing is, no matter what view you approach a problem, the fundamentals and the rules of probability theory remains the same. Hopefully you would choose wisely. Just be a Bayesian. Trust me!



</p>
</section>

<h3 style="color:#2a7ae2"> A brief review of Probability theory </h3>
<section>

<h4 style="color:#2a7ae2"> Discrete Random Variables </h4>

<p>
The following material would make more sense, if you know about random variables. And If not, I guess it should be fine. But definitely put it on your checklist. NOW!
</p>

<p>
A binary event is an event which has only two outcomes like True or False. We can use p(A) to denote probability of a certain event. When p(A) = 1, it means the probability of event A occuring is 1, i.e., A would definitely happen and usually 1 denotes True (by convention). A could be any logical expression, for instance  A could be "It will rain tomorrow". So when p(\(\overline{A}\)) = 0, it basically means it won't rain tomorrow with a probability of 0, where p(\(\overline{A}\)) = 1 - p(A). Easy right?
</p>
<p>
&emsp;&emsp; 
We could extend the idea of these binary events by  defining a discrete finite set or countably infinite set \(\mathcal{X}\) over something called  <strong>discrete random variables rvs (X). </strong> The X could hold any value x (X=x), where x \( \epsilon\) \(\mathcal{X}\). In case an event is binary as stated in the above example, \(\mathcal{X}\) = {0,1} and p("it will rain") = 1 and p("it will not rain") = 0. Random variables are basically a mapping from an event in the real world to a value from this finite space \(\mathcal{X}\).
An event could be represented by a random variable having some value i.e., p(X = x) pr p(x) for short. This function p() is called the <strong>proabability mass function (pmf) </strong> and statisfies certain properties such as 0 \(\leq\) p(X) \(\leq\) 1 and \(\sum_{x \epsilon \mathcal{X}}\) p(x) = 1. 

A non-binary example would be where the finite state space \(\mathcal{X}\) = {1,2,3,4} and probability of a random variable Y having each of these values is equally probable P(Y = x) = 0.25 where x \( \epsilon\) \(\mathcal{X}\).  This is an example of uniform distribution. Dont worry, we will talk about distributions later in the post. 
</p>


<h4 style="color:#2a7ae2"> Fundamental Rules </h4>

<h5 style="color:#2a7ae2"> Probability of Union of Two Events </h5>
<p>
Given two events A and B, the probability of either A or B being True, is the defined as the union of the two events. <br/>
$$ p(A \cup B) = p(A) + p(B) - p(A \cap B) $$
$$p(A \cup B) = p(A) + p(B)  \text{ if A & B are mutually exclusive.} $$
</p>



<h5 style="color:#2a7ae2"> Joint Probability </h5>
<p>
A Joint probability is the likelihood of two events A & B occuring at the same point in time, and is defined as the intersection of the two events.

$$ p(A,B) = p(A \cap B) = p(A|B)p(B) $$
This is also called the <strong>product rule.</strong>
</p>

<p>
Give a joint probability distribution on events A & B we define the <strong>marginal distribution</strong> as follows :
$$ p(A) = \sum_{b} p(A,B) =\sum_{b} p(A|B = b)p(B = b) $$

To calculate the marginal probability of an event A, we sum over all the possible values of other given events, in this case B. This is also called the <strong>sum rule </strong> or <strong>the rule of total probability.</strong>
</p>


<h5 style="color:#2a7ae2"> Conditional Probability </h5>
<p>
We define conditional probability of an event A, given that event B is true or has already occured.
$$ p(A|B) = \frac{p(A,B)}{p(B)} \text{where p(B) > 0} $$
</p>

<h4 style="color:#2a7ae2"> Bayes' Rule </h4>

<p>
{% highlight ruby %}
"Bayes' theorem is to the theory of probability what Pythagoras's theorem is to geometry."
-Sir Harrold Jeffrey's 
{% endhighlight  %}

</p>
<p>
This is one of the most important theories in Probability and is one of the core foundations of Bayesian Machine Learning. And is turns out, it is pretty simple and intuitive. We can define it by building on the sum and product rule definitions as described in the above section.

$$ p(A = a|B = b) = \frac{p(A = a,B = b)}{p(B = b)} = \frac{p(B = b|A = a)p(A = a)}{\sum_{a'}p(A = a')p(B = b|A = a')}$$

Lets discuss a few applications to gain more intuition on this theorem.
</p>

<p>
<strong>Medical Diagnosis : </strong>
Lets consider the following medical problem. Suppose a patient in her late 40s gets a medical test done for breast cancer and she would like to know the probability of her having it. Lets ask a question here. What are we really looking for? Do we want to find out the probability of the test being positive, given she has the cancer or the probability that the test is positive given she has the cancer. Well, we should definitely try to find out the latter because the former just gives us an idea of how reliable or sensitive the test is.
</p>

<p>
Lets define two events : <br>
<b>C</b> = Patient has Cancer <br>
<b>T</b> = Test is Positive <br>
</p>

<p>
Let the sensitivity of the test be 80%. This signifies how reliable the test is, given that patient has cancer. $$ p(T|C) = 0.8 $$ 

This gives an idea to the patient she is 80% likely to have the cancer. But this is not true! We have not considered the prior probability of having breast cancer (which is thankfully quite low) 

$$ p(C) = 0.004 $$
and so we need to conider this <strong>base rate fallacy.</strong> We also need to account for false alarms and therefore consider these false positives in the equation as well, i.e., given she has no cancer and the test is positive.
$$ p(T|\overline{C}) = 0.1 $$

Combining these terms and using the Bayes' rule, we can calculate the probability of the patient having the cancer when the test is positive. 

$$ p(C|T) = \frac{p(T|C)p(C)}{p(T|C)p(C) + p(T|\overline{C})p(\overline{C})} $$
$$ p(C|T) = \frac{0.8 * 0.004}{0.8 * 0.004 + 0.1 * 0.996} = 0.031 $$

where p(\(\overline{C}\)) = 1 - p(C) = 0.996 <br>

In other words, we just found out that even if the patient has tested positive, they still only have 3% chance of actually having the breast cancer. <br>

Therefore recommending the test to the masses can definitely raise a lot of false alarms and may cause unnecessary emotional and financial distress. So we need to find an optimal trade-off risk versus reward in case of uncertainity. 
</p>



<h4 style="color:#2a7ae2"> Independence and Conditional Independence </h4>
<p>

We say two random variables X & Y are <strong>unconditionally independent</strong> or <strong>marginally independent</strong>, denoted by 
X \( \bot \)Y, if the joint probability can be represented as the product of two marginals.

$$ X \bot Y \Leftrightarrow p(X,Y) = p(X)p(Y) $$

So, in general we can say that a given set of random variables are mutually independent if the joint can be written as a set of product of their marginals. It basically means that one random variable doesn't really influence the other one.

</p>

<p>
But in reality, it is rare. Since almost always the random variables in a given system affect each other, and unconditional independence is hardly found in practial scenarios. However, the variables don't influence each other directly but through some mediator variable, giving rise to independence but conditional. We therefore say that the random variables X & Y are <strong>conditionally independent (CI)</strong> given a Z, if we can represent their conditional joint as the product of conditional marginals.

$$ X \bot Y | Z \Leftrightarrow p(X,Y|Z) = p(X|Z)p(Y|Z) $$
</p>

<p>
Conditional Independence basically captures the intuition that all the dependencies between X & Y are mediated through Z. It can sound a bit confusing at first, so lets demonstrate this by a scenario.
For example, we can say that the probability it will rain tomorrow (Event X), is independent of the fact that the ground is wet today(Event Y), given the knowledge of whether it is raining today (Event Z).
Intuitively, this is because "Z" causes both "X" & "Y", and so if we know Z, we don't need to know about Y to predict X or vice versa. Since both X & Y depend on the fact that it rained today. 
</p>





<h4 style="color:#2a7ae2"> Discrete Random Variables </h4>

<p>
In this section, we will see how to extend the concept of probability to reason about uncertain continous quantities. Suppose X is some uncertain continous quantity. The probability that X lies in the interval 
\(a \leq X \leq b\) can be computed in the following manner. Let there be events such as \(A = (X \leq a) \), \(B = (X \leq b) \) and \(C = (a \lt X \leq b) \). We have C = \(A \vee B\) and as we can see that events A and C are mutually exclusive, using the sum rule we can get,

$$ p(B) = p(A) + p(C) \Leftrightarrow p(C) = p(B) - p(A) $$

Now lets define the function \( F(k) \triangleq p(X \leq k)\). This function which represents the probability of continous rv less than a value k, is called <strong>cumulative distribution function </strong> or <strong>cdf</strong> and it is a monotonically non-decreasing function. Now we can represent the above inequality in terms of cdf as,

$$ p(a \lt X \leq b) = F(b) - F(a) $$
</p>

<p>
Assuming that the derivative of this cdf exists, lets define something called <strong>probability distribution function </strong> or <strong>pdf</strong>, which can be denoted as f(x),

$$ f(x) = \frac{d}{dx} F(x) \quad \text{where F(x) is the cdf} $$

Given a dpf, we can compute the probability of a continous rv being in a finite interval [a,b] as follows :
$$ p(a \lt X \leq a) = \int_a^b f(x)dx $$

As the size of the interval gets smaller, we can write it as,

$$ p(a \leq X \leq X + dx) \approx p(x)dx $$

</p>

<p>
Also, it should follow certain contraints such as the value of p(x) \( \geq 0 \), but it can also be p(x) \( \gt 1 \), as long as the pdf integrates to 1. For example, consider the following uniform distribution, 

$$ Unif(x|a,b) = \frac{1}{b-a} I(a \leq X \leq b) $$

If we set a = 0 and b = \(\frac{1}{2}\), the value of p(x) = 2, for any x \(\epsilon  [0, \frac{1}{2}]\).  
</p>
</section>
<p> <b>
	End of Part 1
</b>
</p>
<p> The second part of this post covering continous RVs and distributions will soon be up. </p>



























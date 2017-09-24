---
layout: post
title:  "The World of Dimensionality Reduction"
date:   2017-05-30 01:05:31 -0400
categories: machine learning
permalink: /machinelearning/dimensionality-reduction-part1-vector-spaces

---

<br/>

<h3 style="color:#2a7ae2">  Introduction </h3>

<section>
	<strong> <i>
	In machine learning and statistics, dimensionality reduction or dimension reduction is the  process of reducing the number of random variables under consideration, via obtaining a set of principal variables. It can be divided into feature selection and feature extraction. <br>
										~wikipedia
	</i></strong>
</section>

<br>
<section>
<p>
Dimensionality reduction is a major research/interest area for machine learning scientists/enthusiasts. It is basically a set of mathematical techniques which can use be used to make more sense of data. Imagine your boss at Amazon has given you a huge dataset of customer's buying habbits and activities on the website, and has asked you make sense of this huge pile of data being generated in process. Now there are multiple ways to go about it depending upon the exact outcome of the problem. For instance, lets say your're not really sure of how you'd like to approach the problem and you decide to apply some unsupervised learning techniques on your data, to organize it into clusters and groups in order to better understand the patterns. But, often times it could be hard if you're not really sure of what are you looking for, and you might need to make assumptions you aren't really comfortable with. We don't need to go into details about unsupervised learning algorithms today, lets save the best for the last. Before that its a good idea to first understand what really is dimensionality reduction and the intuition behind it.
</p>
<p> 
As the name suggests, its a process to reduce the dimensions in the data. Lets go back to our Amazon example, where lets say the dataset given by your boss has 100 dimensions. This basically means each customer in the dataset, has 100 different fields or values assocaited with him like lets say location, age, gender etc. But when you consider all these dimensions or fields at the same time, it doesn't really tell you much, in the 100-dimension space. So its a good idea, to bring the dimensions down so that you can better understand your customers and try to identify the various degrees of freedom or the modes of variability. Its often a good idea to undertand the covariance of the data and try to identify the ways in which your data can high variance. This may all sound a little not too-familiar, so lets try to learn a few things before we dive deep into it. A good idea is to first understand what are vector spaces. If you're familiar with it, its okay to skip this post and move on to where we start discussing various dimensionality reduction techniques. 

</p>

<p> Before we proceed to discuss about Vector spaces, remember Dimensionality reduction is your friend and its an incredible tool to have in your arsenal to better understand and visualize your data, be it images or just news articles. Over the course of next few posts, we would discover various papers that were landmark in the field of dimensionality reduction and how exactly they acheive what they achieve. I'll try to make the post not math heavy, but I'll surely include all the proofs and mathematics that should help you understand the intuition. 
If you aren't familiar with vector spaces, I highly suggest to go through this brief post before we move on to the algorithms. Trust me, it'll make your life easier. So without further ado, lets checkout this beautiful world of vector spaces. 
</p>
</section>

<br>
<h3 style="color:#2a7ae2">  Vector Space </h3>
<section>
<p>
<img src="/assets/images/machine-learning/vector-space.jpg">
<br><br>
A vector space V is a collection (bag) of objects (vectors) that is closed under operations such as object (vector) addition and scalar (field F) multiplication, and it also needs to satisfy certain properties or axioms.
</p>

<p>
<b> Axioms : </b> <br>


1) \( (\alpha + \beta)v = \alpha v + \beta v, \forall v \in \text{V and } \alpha, \beta \in F \) <br>
2) \(  \alpha(\beta v) = (\alpha\beta)v   \) <br>
3) \(  u + v = v + u \text{ , where  u,v } \epsilon V \) <br>
4) \(  u + (v + w) = (u + v) + w \text{ , where  u,v,w } \epsilon V \)  <br>
5) \( \alpha(u + v) = \alpha u + \alpha v   \) <br>
6) \(  \exists 0 \; \epsilon V \; \Rightarrow 0 + v = v \text{ ; 0 is usually called the origin or zero vector}  \) <br>
7) \(  0 v = 0  \) <br>
8) \(  e v = v \text{ , where e is the multiplicative unit in F}  \) <br>

</p>

<p> Vectors in n-dimensions space, can be represented as n-dimensional tuple.<br>
	For example,  \(R^n = {(a_1,a_2,...,a_n)|a_1,a_2,...a_n \; \epsilon \; R}\)
</p>
</section>

<h4 style="color:#2a7ae2">  Subspace </h4>
<section>
	<p>
		Let V be a vector space and U \( \subset \) V, we will call U a subspace of V, if U is also closed under vector addition, scalar multiplication and satisfies all other vector axioms.
	</p>
	<p>
		For example, V = \(R^3 = {(a, b, c)|a, b, c \; \epsilon \; R}\) <br>
		U = \({(a, b, 0)|a, b \; \epsilon \; R}\) <br>
		Clearly U \( \subset \) V, and U is a subspace of V.
	</p>
	<p>
		Let \( v_1,v_2 \epsilon R^3\) and W = \( {a v_1 + b v_2 | a, b \epsilon R} \)
		And W is a subspace of \( R^3\). <br>
		In this case, we say W is "spanned" by \( {v_1,v_2} \) <br>

	Okay, lets understand what is a span first.

</p>
</section>

<h4 style="color:#2a7ae2">  Span </h4>
<section>
	<p> In general, let S \( \subset \) V, where V is a vector space,and S has the form, S = \({v_1, v_2,...,v_k} \)
	</p>
	<p>The span of the set S is the set U, where U \( \subset \) V. 
		$$ U = { \sum_{j=1}^k a_j v_j | a_1,..., a_k \epsilon \;R } $$  Span is sometimes denoted like \( \mathcal{G}(v_1, v_2,...,v_k) \)</p>

<p> <b>Linear Combination of Vectors</b><br> 
We say any vector (say u) is a linear combination of vectors (say \( v_1, v_2,..., v_3 \)), if it can be represented as \( u = a_1 v_1 + a_2 v_2 + ... + a_k v_k \text{ where} a_1,...,a_k are fields \).
</p>
</section>

<h4 style="color:#2a7ae2">  Linear Dependence </h4>

<section>
<p>
Let S = {\( v_1, v_2,..., v_k \)} \(\subset\) V, a vector space. We say that S is linearly dependent if 
<b>$$  a_1 v_1 + a_2 v_2  + ... + a_k v_k =  0 \text{ where not all coefficients are 0}$$ </b>

As you can see that any vector in this space can be represented as a linear combination of other vectors, therefore it has linear dependence <b>(NOT LINEARLY INDEPENDENT). </b>
</p>	
</section>

<h4 style="color:#2a7ae2">  Basis </h4>
<section>
<p>
The idea is to find a minimal generating set for a vector space. 
</p>
<p>
Basically suppose V be a vector space, and S = {\( v_1, v_2, ..., v_k\)} is a linearly independent spanning set for V. Then <u>S is called a basis of V</u>.
</p>	
<strong>Proposition : </strong> If S is a basis of V, then every vector has a unique representation. <br><br>
For example, <br>
Let V = \( R^3 \), and S = {\(e_1, e_2, e_3\)}, where \(e_1, e_2, e_3\) are the multiplicative units of V. Then S is a basis for V.

<p>
<b>Proof: </b> <br>
Clearly, V is spanned by S, as any vector in this vector space can be represented in the form of S. Important thing is to find out if S is linearly independent. 
<br>

$$ 0 =  a_1 e_1 + a_2 e_2 + a_3 e_3 $$ 
$$ (0, 0, 0) =  a_1 (1, 0, 0) + a_2 (0, 1, 0) + a_3 (0, 0, 1)  $$
$$ (0, 0, 0) =  (a_1, a_2, a_3) $$

<strong> Hence, its linearly independent since, it is only 0, when all the coefficients together are 0,</strong> i.e., \( a_1=0, a_2=0, a_3=0 \)
</p>
</section>

<h4 style="color:#2a7ae2">  Dimension </h4>
<section>
<p>
	The dimension of a vector space V is the number vectors in a basis of V.
</p>
<p>
<b>Theorem: </b> <br>
Let S = {\( v_1, v_2,..., v_k\)} \( \subset \) V, be a basis for V, then every basis of V has k elements. <br>

$$dim(R^n) = n $$
$$dim(P^n) = n + 1$$

<br>
<b>Example:</b>
Let \( P^n = \) {\(a_n x^n + a_{n-1} x^{n-1} + ... + a_1 x^1 + a_0 x^0\)} be the vector space of  polynomials of degree n.
And so the powers <\( x^0, x^1, ..., x^n \)> they are linearly indpendent and they form a basis of \( P^n \).
$$ P^n = \mathcal{G}(S)  \text{ , where S = }(1, x, ..., x^n) $$
</p>
</section>

<h4 style="color:#2a7ae2">  Norms </h4>
<section>
<p>
	It is a way of putting a measure of distance on vector space. <br>
Suppose that \( \left\|.\right\| \; V \rightarrow R^{+}\) is a function from V to the non-negative reals for which it needs to satisfy certain axioms.
</p>
<p>
<b>Axioms:</b>

1) \(  \left\|v\right\| > 0 \; \forall v \in \text{V and } \left\|v\right\| = 0 \text{ if and only if } v = 0 \) <br>
2) \(  \left\| \alpha v\right\| = \alpha \left\| \ v\right\| \; \forall \alpha \in C, R and v \in V \) <br>
3) \(  \left\| u + v \right\|  \leq \left\| u\right\| + \left\| v\right\| \forall v \in V\). This is the <b>"triangular inequality"</b> <br>
</p>
<br>
<p>
Let V = \( R^n (or C^n) \). Define for x = (\( x_1, x_2, ..., x_n \) ) <br>
1) <strong>l2 norm : </strong>  \( \left\|x\right\|_2 = (\left\lvert x_1 \right\rvert^2 + \left\lvert x_2 \right\rvert^2 + ... + \left\lvert x_n \right\rvert^2 )^{1/2} \) <br>
2) <strong>l1 norm : </strong> \( \left\|x\right\|_1 = (\left\lvert x_1 \right\rvert + \left\lvert x_2 \right\rvert + ... + \left\lvert x_n \right\rvert ) \) <br>
3) \( \left\|x\right\|_\infty = \max{1 \leq i \leq n} \; ( l_{\infty} norm )\) 
</p>	
</section>

<h4 style="color:#2a7ae2">  Inner Product Space </h4>
<section>
<p>
<strong> Inner Product </strong> <br>
	Let F be the field, reals or complex numbers and V be a vector space over F. An inner product on V is a function, 
	$$ (*, *) : v \; x \; v \; \rightarrow F $$

1) (au + bv , cw) = a(u,w) + b(v,w)  \( \forall a,b \in F and \forall \; u, v, w \in V \) <br>
2) (u, v) =  \( (\overline{u, v}) \; \forall u, v \in V \) <br>
3) (v,v) > 0 , \( \forall \; \text{non-zero v} \in  \) V <br>
4) Also, define\( \left\|v\right\| = \sqrt{(v,v)} \) <br>
</p>
<br>
<p>
<strong> Inner Product Space </strong>
A vector space V over F with an inner product (*,*) is said to be an inner product space. Also knows as Hilber Space. <br>
An inner product space V over Reals R is known as Euclidean Space whereas V over complex numbers C is known as unitary space.
</p>
<p>
Again, like vector spaces, it needs to satisy certain axioms. <br>
Let F = R or C, and V be an inner product space over F. For v,w \( \in V \) and C \( \in \) F, we have : <br>
1) \(  \left\|c v\right\|  = \left\lvert c \right\rvert \left\|v\right\| \) <br>
2) \( \left\|c v\right\| > 0, \text{if v} \neq 0 \) <br>
3) \( \left\lvert (u, v) \right\rvert  \leq \left\|u\right\| \left\|v\right\| \) <br>
This equality only holds true if and only if \(  v = \frac{(u,v)}{\left\|v\right\|^2} v\). Also known as <b>"Cauchy'Schwartz inequality"</b> <br>
4)\( \left\|u + v\right\|  \leq \left\|u\right\| + \left\|v\right\|\). <br>Triangular Inequality holds true in Hilbert space as well.
</p> 
 </section>

<h4 style="color:#2a7ae2">  Orthogonality </h4>
<section>
<p>
Let V be an inner product space over the field F,  <br>
1) Suppose v,w \( \in \) V, then v & w are mutually orthogonal if the inner product |v,w| = 0 <br>
2) A subset S \( \in V\) is said to be an orthogonal set if v \( \perp \) w, \( \forall \) v, w \( \in \) S, and \( v \neq w \). <br>
3) An orthogonal set S is said to be an orthogonal set if \( \left\|v\right\| = 1, \; \forall \; v \in \; S\)<br>
4) Zero Vector is orthogonal to all elements of V.

</p>
<br>
<p>
<strong>Theorem 1: </strong> <br>
Let V be an inner product space over F, and S be an orthogonal set of non-zero vectors. Then S is linearly indpependent. <br>
Let V = \(R^n \) or V =\(C^n \). The the standard basis E = {\( e_1, e_2, ..., e_3\)} is an orthornormal set.
</p>

<br>
<strong>Theorem 2: </strong> <br>
Let V be an inner product space over F, and let <u> \( v_1,v_2,..., v_k \)</u> be linearly independent set of vectors. Then we can construct elements <u>\( e_1, e_2, ..., e_k \; \in \) V </u> such that  <br>
1) \( e_1, e_2, ..., e_k \) is an orthonormal set
Let V = \(R^n \) or V =\(C^n \). The the standard basis E = {\( e_1, e_2, ..., e_3\)} is an orthornormal set. <br>
2) \( e_k \; \in \) span({\( v_1, v_2,...v_k \)}) <br>

<b>The proof of this theorem is known as Graham Schmidt Orthogonalization Process.</b>
</p>
</section>

<br>
So, this was a very brief introduction to Vector & Inner Product Spaces. I highly recommend that you study further (Wikipedia is a great source for metric spaces), as it will only allow you to better understand how mathematics and its abstractions play such an important role in machine learning. But in case, you aren't interested, it should be enough for you to understand the various dimensionality reduction techniques that we are going to talk about in the next few posts. See ya!




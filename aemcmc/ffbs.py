from typing import Mapping, Tuple

import aesara
import aesara.tensor as at
from aesara.tensor.random import RandomStream
from aesara.tensor.var import TensorVariable


def ffbs_step(
    gamma_0: TensorVariable,
    Gammas: TensorVariable,
    log_lik: TensorVariable,
    srng: RandomStream,
    lower_prec_bound: float = 1e-20,
) -> Tuple[TensorVariable, Mapping]:
    r"""Draw a discrete state sequence sample using forward-filtering backward-sampling (FFBS) [fs]_.

    FFBS draws samples according to

    .. math::

        S_T &\sim p(S_T | y_{1:T}) \\
        S_t \mid S_{t+1} &\sim p(S_{t+1} | S_t) p(S_{t+1} \mid y_{1:T})

    for discrete states in the sequence :math:`S_t`, :math:`t \in \{0, \dots, T\}`
    and observations :math:`y_t`.

    The argument `gamma_0` corresponds to :math:`S_0`, `Gammas` to
    :math:`p(S_{t+1} | S_t)`, and `log_lik` to :math:`\log p(y_t \mid S_t)`.
    The latter is used to derive the forward quantities
    :math:`p(S_{t+1} \mid y_{1:T})`.

    This implementations is most similar to the one in [nr]_, which forgoes
    expensive log-space calculations by using fixed precision bounds and
    re-normalization/scaling.

    Parameters
    ----------
    gamma_0
        The initial state probabilities as an array of base shape ``(M,)``.
    Gamma
        The transition probability matrices.  This array should take the base
        shape ``(N, M, M)``, where ``N`` is the state sequence length and ``M``
        is the number of distinct states.  If ``N`` is ``1``, the single
        transition matrix will broadcast across all elements of the state sequence.
    log_lik
        An array of base shape ``(M, N)`` consisting of the log-likelihood
        values for each state value at each point in the sequence.
    lower_prec_bound
        A number indicating when forward probabilities are too small and need
        to be re-normalized/scaled.

    Returns
    -------
    A tuple containing the tensor representing the FFBS sampled states and the
    `Scan`-generated updates.

    References
    ----------

    .. [fs] Sylvia Fruehwirth-Schnatter, "Markov chain Monte Carlo estimation of classical and dynamic switching and mixture models", Journal of the American Statistical Association 96 (2001): 194--209
    .. [nr] Press, William H., Saul A. Teukolsky, William T. Vetterling, and Brian P. Flannery. 2007. "Numerical Recipes 3rd Edition: The Art of Scientific Computing." Cambridge university press.

    """

    gamma_0 = at.as_tensor(gamma_0)
    log_lik = at.as_tensor(log_lik)

    # Number of observations
    N = log_lik.shape[-1]

    # Number of states
    # M = gamma_0.shape[-1]

    # Initial state probabilities
    gamma_0_normed = gamma_0
    gamma_0_normed /= at.sum(gamma_0)

    # Make sure we have a transition matrix for each element in a state
    # sequence
    Gammas = at.broadcast_to(Gammas, (N,) + Gammas.shape[-2:])

    def forward_step(log_lik_T_n, Gamma_n, alpha_nm1):
        log_lik_n = log_lik_T_n.T
        lik_n = at.exp(log_lik_n - log_lik_n.max())
        alpha_n = at.dot(alpha_nm1, Gamma_n) * lik_n
        alpha_n_sum = at.sum(alpha_n, axis=-1)
        rescaled_alpha_n = at.switch(
            at.lt(alpha_n_sum, lower_prec_bound), alpha_n / lower_prec_bound, alpha_n
        )
        return rescaled_alpha_n

    alphas, _ = aesara.scan(
        forward_step,
        sequences=[log_lik.T, Gammas],
        outputs_info=[{"initial": gamma_0_normed, "taps": [-1]}],
        non_sequences=[],
        n_steps=N,
        strict=True,
        name="forward-pass",
    )

    alphas = alphas.T

    def backward_step(alphas_T_n, Gamma_np1, state_n):
        alphas_n = alphas_T_n.T
        beta_n = alphas_n * Gamma_np1[:, state_n]
        beta_n = beta_n / at.sum(beta_n, axis=-1)
        state_np1 = srng.categorical(beta_n)
        return state_np1

    alpha_N = alphas[..., N - 1]
    beta_N = alpha_N / alpha_N.sum(axis=-1)

    state_np1 = srng.categorical(beta_N)

    state_samples_rev, updates = aesara.scan(
        backward_step,
        sequences=[alphas.T[: N - 1], Gammas[1:]],
        outputs_info=[{"initial": state_np1, "taps": [-1]}],
        non_sequences=[],
        go_backwards=True,
        strict=True,
        name="backward-pass",
    )

    state_samples_rev = at.join(
        0, at.atleast_Nd(state_np1, n=state_samples_rev.type.ndim), state_samples_rev
    )

    return state_samples_rev[::-1], updates

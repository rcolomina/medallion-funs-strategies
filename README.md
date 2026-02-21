# Purpose

The purpose of this repository is to gather the known mathematics behind the trading techniques of the Medallion Fund run by Reinassance Technologies

## Medallion Fund

The Medallion Fund, run by Renaissance Technologies (founded by mathematician Jim Simons), is widely considered the most successful hedge fund in history. Here's what's known about it:

The fund averaged roughly 66% annual returns before fees (about 39% after fees) from 1988 through the 2010s. It has had almost no losing years, even profiting during major market crashes like 2008. The fund has been closed to outside investors since 1993 — only Renaissance employees can invest.

## Hidden Markov Models

This is the foundational framework. Simons and his researchers used the Baum-Welch algorithm to determine the parameters of Hidden Markov Models applied to historical financial data. Day-to-data The key insight was that financial markets can be modeled as hidden Markov chains — sequences of events where the probability of what happens next depends only on the current state, not past events, but the underlying states driving those events are not directly observable. Medium

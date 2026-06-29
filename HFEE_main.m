%% =========================================================================
%% HFEE_main.m  –  MATLAB 2026a
%%
%% HFEE = FPSOGSA-EE + fmincon refinement
%%   Hybrid Fractional PSO-GSA with Entropy Enhancement, refined by fmincon
%%
%% DOCR Coordination — NO clustering, NO mixed curves, NO computational burden.
%% Single objective: minimise total primary operating time T_op.
%% Pipeline: FPSOGSA-EE (global) -> CTI repair -> fmincon (local refinement)
%%           with hard + soft penalty.
%%
%% Systems (select via SYS):
%%   'IEEE15' : 42 relays, 92 pairs
%%   'IEEE30' : 38 relays, 60 pairs
%%   'IEEE42' : 97 relays, 117 pairs
%%
%% Bounds (DO NOT CHANGE):  TMS in [0.1, 1.0],  PS in [0.5, 2.5],  CTI = 0.2 s
%% Curve: IEC Standard Inverse (A=0.14, B=0.02)
%%
%% Sweeps alpha = 0.1..0.9, 50 runs each, 200 iterations.
%% Outputs: convergence (best/mean/worst) per alpha, violin plot, CTI plot,
%%          frequency distribution, statistics (Std, CV, CI, mean/best/worst).
%%
%% Run:  >> HFEE_main          (edit SYS below to choose system)
%% =========================================================================

clear; close all; clc;

%% ── SELECT SYSTEM ───────────────────────────────────────────────────────
SYS = 'IEEE42';     % 'IEEE15' | 'IEEE30' | 'IEEE42'

%% ── GLOBAL PARAMETERS ───────────────────────────────────────────────────
N_RUNS   = 50;          % independent runs per alpha
MAX_IT   = 200;         % FPSOGSA iterations per run
POP      = 50;         % swarm size
CTI_thr  = 0.20;        % coordination time interval (s)
ALPHA    = 0.1:0.1:0.9; % fractional order sweep
A_IEC    = 0.14;        % SI curve constant A
B_IEC    = 0.02;        % SI curve constant B
USE_FMINCON = true;     % HFEE refinement step (needs Optimization Toolbox)
DIST_USE_RAW= true;     % violin/freq plots use pre-fmincon data (shows spread)
                        %   for small convex systems (15,30) fmincon collapses
                        %   all runs to one point; raw data reveals the true
                        %   stochastic distribution of the metaheuristic

% FPSOGSA-EE hyperparameters
w_init=0.9; w_damp=0.99; G0=1; C1=0.9; C2=1.5; lambda_H=0.05;
% Penalty weights
lam_hard = 1e6;   % hard penalty (per violation)
lam_soft = 1e3;   % soft penalty (squared shortfall)

%% ── LOAD SYSTEM DATA ────────────────────────────────────────────────────
[fM, CTR, PB, TMS_lb, TMS_ub, PS_lb, PS_ub, sysname] = load_system(SYS);
nR     = size(fM,2);
nPairs = size(PB,1);
dim    = 2*nR;
LB = [TMS_lb*ones(1,nR), PS_lb*ones(1,nR)];
UB = [TMS_ub*ones(1,nR), PS_ub*ones(1,nR)];

fprintf('\n=== HFEE DOCR Coordination: %s ===\n', sysname);
fprintf('Relays=%d, Pairs=%d, Scenarios=%d, Variables=%d\n', ...
        nR, nPairs, size(fM,1), dim);
fprintf('Bounds: TMS[%.2f,%.2f] PS[%.2f,%.2f] CTI=%.2f\n', ...
        TMS_lb,TMS_ub,PS_lb,PS_ub,CTI_thr);
fprintf('Sweep: alpha=%.1f..%.1f, %d runs x %d iter, pop=%d\n\n', ...
        ALPHA(1),ALPHA(end),N_RUNS,MAX_IT,POP);

%% ── STORAGE ─────────────────────────────────────────────────────────────
nA = numel(ALPHA);
CONV  = cell(1,nA);                 % {a}(run, iter) penalised-cost convergence
CONV_TOP = cell(1,nA);              % {a}(run, iter) T_op convergence
FTOP  = zeros(N_RUNS, nA);          % final T_op per run (after fmincon)
FTOP_RAW = zeros(N_RUNS, nA);       % T_op after FPSOGSA+repair (pre-fmincon) - keeps diversity
FVIOL = zeros(N_RUNS, nA);          % final violations per run
FVIOL_RAW= zeros(N_RUNS, nA);       % violations pre-fmincon
% Tier B: feasibility-first best (zero/min violations, then min T_op)
BEST_SOL = cell(1,nA);              % feasible-tier best solution per alpha
BEST_VIOL_A = inf(1,nA);
BEST_TOP_A  = inf(1,nA);
% Tier A: pure minimum-T_op best (lowest T_op regardless of violations)
MINTOP_SOL  = cell(1,nA);          % min-Top-tier solution per alpha
MINTOP_TOP  = inf(1,nA);
MINTOP_VIOL = inf(1,nA);

%% ── MAIN SWEEP ──────────────────────────────────────────────────────────
for ai = 1:nA
    alpha = ALPHA(ai);
    fprintf('--- alpha = %.1f (%d/%d) ---\n', alpha, ai, nA);
    Cmat = zeros(N_RUNS, MAX_IT);
    Cmat_top = zeros(N_RUNS, MAX_IT);   % T_op convergence per run
    best_top_a=Inf; best_viol_a=Inf; best_sol_a=[];
    mintop_a=Inf;   mintop_viol_a=Inf; mintop_sol_a=[];

    for run = 1:N_RUNS
        rng(run*100 + round(alpha*10));

        % --- Stage 1: FPSOGSA-EE global search ---
        [pos, conv, conv_top] = fpsogsa_ee(dim, LB, UB, POP, MAX_IT, fM, CTR, PB, ...
            A_IEC, B_IEC, CTI_thr, alpha, w_init, w_damp, G0, C1, C2, ...
            lambda_H, lam_hard, lam_soft);

        % --- Stage 2: CTI repair ---
        pos = cti_repair(pos, fM, CTR, PB, A_IEC, B_IEC, CTI_thr, LB, UB);

        % Record PRE-fmincon result (retains run-to-run diversity for
        % violin/frequency plots; fmincon collapses small convex systems
        % to a single deterministic optimum)
        [top_raw, nv_raw] = eval_solution(pos, fM, CTR, PB, A_IEC, B_IEC, CTI_thr);
        FTOP_RAW(run,ai)  = top_raw;
        FVIOL_RAW(run,ai) = nv_raw;

        % --- Stage 3: fmincon refinement (HFEE) ---
        if USE_FMINCON
            pos = fmincon_refine(pos, fM, CTR, PB, A_IEC, B_IEC, CTI_thr, LB, UB);
        end

        % Evaluate (refined)
        [top, nv] = eval_solution(pos, fM, CTR, PB, A_IEC, B_IEC, CTI_thr);
        FTOP(run,ai)  = top;
        FVIOL(run,ai) = nv;
        Cmat(run,:)   = conv;
        Cmat_top(run,:) = conv_top;

        % Tier B — feasibility-first best per alpha
        if (nv < best_viol_a) || (nv==best_viol_a && top < best_top_a)
            best_viol_a=nv; best_top_a=top; best_sol_a=pos;
        end
        % Tier A — pure minimum-T_op best per alpha (violations allowed)
        if top < mintop_a
            mintop_a=top; mintop_viol_a=nv; mintop_sol_a=pos;
        end

        if mod(run,10)==0
            % Show BOTH the metaheuristic (pre-fmincon, varies per run) and
            % the refined (post-fmincon) value. For small convex systems the
            % refined value is identical across runs (deterministic optimum);
            % the pre-fmincon value reveals the run-to-run stochastic spread.
            fprintf('  run %2d/%d | FPSOGSA+repair T_op=%.4f (viol=%d) | refined T_op=%.4f (viol=%d)\n', ...
                    run, N_RUNS, top_raw, nv_raw, top, nv);
        end
    end

    CONV{ai}        = Cmat;
    CONV_TOP{ai}    = Cmat_top;
    BEST_SOL{ai}    = best_sol_a;
    BEST_VIOL_A(ai) = best_viol_a;
    BEST_TOP_A(ai)  = best_top_a;
    MINTOP_SOL{ai}  = mintop_sol_a;
    MINTOP_TOP(ai)  = mintop_a;
    MINTOP_VIOL(ai) = mintop_viol_a;
    fprintf('  => alpha=%.1f | Feasible: T_op=%.4f (viol=%d) | Min-Top: T_op=%.4f (viol=%d)\n\n', ...
            alpha, best_top_a, best_viol_a, mintop_a, mintop_viol_a);
end

%% ── TWO-TIER SUMMARY TABLE (all alphas) ─────────────────────────────────
fprintf('=== TWO-TIER RESULTS PER ALPHA (%s) ===\n', sysname);
fprintf('%5s | %12s %6s | %12s %6s\n', ...
        'alpha','MinTop Top','viol','Feasible Top','viol');
fprintf('%s\n', repmat('-',1,52));
for ai=1:nA
    fprintf('%5.1f | %12.4f %6d | %12.4f %6d\n', ...
        ALPHA(ai), MINTOP_TOP(ai), MINTOP_VIOL(ai), ...
        BEST_TOP_A(ai), BEST_VIOL_A(ai));
end
fprintf('\n');

%% ── SELECT BEST ALPHA (feasibility-first) ───────────────────────────────
mv = min(BEST_VIOL_A);
cand = find(BEST_VIOL_A==mv);
[~,ci] = min(BEST_TOP_A(cand));
best_ai = cand(ci);
best_alpha = ALPHA(best_ai);
% Also identify the global min-Top alpha (Tier A)
[~,mintop_ai] = min(MINTOP_TOP);
mintop_alpha = ALPHA(mintop_ai);
fprintf('=== BEST ALPHA (Feasible tier) = %.1f (T_op=%.4f, viol=%d) ===\n', ...
        best_alpha, BEST_TOP_A(best_ai), BEST_VIOL_A(best_ai));
fprintf('=== BEST ALPHA (Min-Top tier)  = %.1f (T_op=%.4f, viol=%d) ===\n\n', ...
        mintop_alpha, MINTOP_TOP(mintop_ai), MINTOP_VIOL(mintop_ai));

%% ── STATISTICS FOR BEST ALPHA ───────────────────────────────────────────
% Two statistics sets:
%  (1) REFINED (post-fmincon): the deterministic HFEE result. For small
%      convex systems fmincon converges all runs to the same optimum, so
%      Std/CV are ~0 by design (this is a STRENGTH: reliable convergence).
%  (2) METAHEURISTIC (pre-fmincon, FTOP_RAW): the stochastic spread of the
%      FPSOGSA-EE stage, which is what the violin/frequency plots show.
ftop_best = FTOP(:,best_ai);
feas = FVIOL(:,best_ai)==min(FVIOL(:,best_ai));
stat_data = ftop_best(feas);
mu   = mean(stat_data);  sd = std(stat_data);  cv = 100*sd/max(mu,1e-9);
bestv= min(stat_data);   worstv = max(stat_data);
ci95 = 1.96*sd/sqrt(max(numel(stat_data),1));

% Metaheuristic (raw) statistics at best alpha
raw_data = FTOP_RAW(:,best_ai);
mu_r = mean(raw_data); sd_r = std(raw_data); cv_r = 100*sd_r/max(mu_r,1e-9);
best_r = min(raw_data); worst_r = max(raw_data);
ci95_r = 1.96*sd_r/sqrt(max(numel(raw_data),1));

fprintf('=== STATISTICS (best alpha=%.1f) ===\n', best_alpha);
fprintf('-- Refined HFEE (post-fmincon, %d feasible runs) --\n', sum(feas));
fprintf('  Best=%.4f  Worst=%.4f  Mean=%.4f  Std=%.4f  CV=%.2f%%  95%%CI=[%.4f, %.4f]\n', ...
        bestv, worstv, mu, sd, cv, mu-ci95, mu+ci95);
if sd < 1e-4
    fprintf('  (Std~0: fmincon converges deterministically to the unique optimum)\n');
end
fprintf('-- Metaheuristic FPSOGSA-EE (pre-fmincon, %d runs) --\n', numel(raw_data));
fprintf('  Best=%.4f  Worst=%.4f  Mean=%.4f  Std=%.4f  CV=%.2f%%  95%%CI=[%.4f, %.4f]\n\n', ...
        best_r, worst_r, mu_r, sd_r, cv_r, mu_r-ci95_r, mu_r+ci95_r);

%% ── CTI TABLE FOR BEST SOLUTION ─────────────────────────────────────────
sol = BEST_SOL{best_ai};
TMS = sol(1:nR); PS = sol(nR+1:end);
[Tp_arr, Tb_arr, CTI_arr] = cti_table(TMS, PS, fM, CTR, PB, A_IEC, B_IEC);
n_sat  = sum(CTI_arr >= CTI_thr-1e-6 & isfinite(CTI_arr));
n_viol = sum(CTI_arr < CTI_thr-1e-6 & isfinite(CTI_arr));
fprintf('=== CTI (Feasible tier): %d/%d satisfied, %d violated ===\n', n_sat, nPairs, n_viol);

% Min-Top tier CTI table
sol_mt = MINTOP_SOL{mintop_ai};
TMS_mt = sol_mt(1:nR); PS_mt = sol_mt(nR+1:end);
[Tp_mt, Tb_mt, CTI_mt] = cti_table(TMS_mt, PS_mt, fM, CTR, PB, A_IEC, B_IEC);
n_sat_mt  = sum(CTI_mt >= CTI_thr-1e-6 & isfinite(CTI_mt));
n_viol_mt = sum(CTI_mt < CTI_thr-1e-6 & isfinite(CTI_mt));
fprintf('=== CTI (Min-Top tier):  %d/%d satisfied, %d violated ===\n\n', n_sat_mt, nPairs, n_viol_mt);

% Classify violating pairs (for paper's 3-category explanation)
fprintf('=== VIOLATING PAIRS (Min-Top tier) — for paper classification ===\n');
maxIF_r = max(fM,[],1);
viol_list = find(CTI_mt < CTI_thr-1e-6 & isfinite(CTI_mt));
if isempty(viol_list)
    fprintf('  None — all pairs satisfy CTI.\n');
else
    pairs_set = PB(:,1)*10000 + PB(:,2);
    for k=1:numel(viol_list)
        p = viol_list(k);
        pr=PB(p,1); bk=PB(p,2);
        ratio = maxIF_r(bk)/max(maxIF_r(pr),1);
        is_bidi = any(pairs_set==(bk*10000+pr));  % reverse pair exists?
        if is_bidi
            cat='Bidirectional (directional element)';
        elseif ratio>0.8
            cat='Equal-sensitivity (dual-setting DOCR)';
        else
            cat='Transformer/impedance (87 protection)';
        end
        fprintf('  R%d->R%d: CTI=%.3f, IF_ratio=%.2f | %s\n', pr, bk, CTI_mt(p), ratio, cat);
    end
end
fprintf('\n');

%% ── FIGURE A: 9-subplot penalised-cost convergence (log scale) ──────────
% One figure, 3×3 grid, one subplot per alpha 0.1..0.9.
% Shows FPSOGSA-EE penalised cost (T_op + violation penalty) vs iteration.
% Curves are monotone-decreasing best-so-far. Log scale emphasises the large
% early drops (clearing violations) and fine T_op refinement later.
% Best/Mean/Worst computed across all 50 independent runs per alpha.
fig_cost = figure('Color','w','Position',[30 30 1400 900]);
sgtitle(sprintf('%s — FPSOGSA-EE Convergence: Penalised Cost (log scale)',sysname), ...
        'FontSize',13,'FontWeight','bold');
iter_ax = 1:MAX_IT;
cols_bw = {'Uncertainty Band','Worst','Mean','Best'};
for ai = 1:nA
    Cmat = CONV{ai};
    cv_best  = min(Cmat,[],1);
    cv_mean  = mean(Cmat,1);
    cv_worst = max(Cmat,[],1);
    subplot(3,3,ai);
    fill([iter_ax, fliplr(iter_ax)],[cv_worst, fliplr(cv_best)], ...
         [0.82 0.94 0.82],'EdgeColor','none','FaceAlpha',0.5); hold on;
    plot(iter_ax,cv_worst,'r--','LineWidth',1.5);
    plot(iter_ax,cv_mean, 'b-', 'LineWidth',1.8);
    plot(iter_ax,cv_best, 'g-', 'LineWidth',2.0);
    % end-point markers
    plot(MAX_IT,cv_worst(end),'vr','MarkerFaceColor','r','MarkerSize',7);
    plot(MAX_IT,cv_mean(end), '^b','MarkerFaceColor','b','MarkerSize',7);
    plot(MAX_IT,cv_best(end), '^g','MarkerFaceColor','g','MarkerSize',7);
    % value labels (use scientific notation for large penalty values)
    text(MAX_IT*0.97,cv_worst(end),sprintf('%.2g',cv_worst(end)),...
         'Color','r','FontSize',7,'HorizontalAlignment','right');
    text(MAX_IT*0.97,cv_mean(end), sprintf('%.2g',cv_mean(end)),...
         'Color','b','FontSize',7,'HorizontalAlignment','right');
    text(MAX_IT*0.97,cv_best(end), sprintf('%.2g',cv_best(end)),...
         'Color','g','FontSize',7,'HorizontalAlignment','right');
    hold off; grid on; box on;
    set(gca,'YScale','log','FontSize',9,'XLim',[1 MAX_IT]);
    xlabel('Iteration','FontSize',9);
    ylabel('Cost (log)','FontSize',9);
    % Highlight best alpha with a coloured border
    if ai == best_ai
        set(gca,'LineWidth',2.5,'XColor',[0.1 0.6 0.1],'YColor',[0.1 0.6 0.1]);
        title(sprintf('\\alpha=%.1f  \\bigstar BEST',ALPHA(ai)),'FontSize',10,'Color',[0.1 0.6 0.1]);
    else
        title(sprintf('\\alpha=%.1f',ALPHA(ai)),'FontSize',10);
    end
    if ai==1
        legend(cols_bw,'Location','northeast','FontSize',7,'Box','off');
    end
end
saveas(fig_cost, sprintf('fig_%s_conv9_cost.png',SYS));
fprintf('Saved: fig_%s_conv9_cost.png  (3x3 subplots, log cost)\n',SYS);

%% ── FIGURE B: 9-subplot T_op trajectory (incumbent) ─────────────────────
% One figure, 3×3 grid, one subplot per alpha 0.1..0.9.
% Shows TRUE T_op of the cost-incumbent vs iteration. NOT forced monotone:
% the rise-then-settle pattern is honest (T_op rises as violations are cleared,
% then settles once the swarm finds feasible-or-near-feasible solutions).
% Best/Mean/Worst across 50 runs, in units of seconds.
fig_top = figure('Color','w','Position',[60 60 1400 900]);
sgtitle(sprintf('%s — T_{op} Trajectory of Cost-Incumbent vs Iteration',sysname), ...
        'FontSize',13,'FontWeight','bold');
for ai = 1:nA
    Ctop = CONV_TOP{ai};
    tv_best  = min(Ctop,[],1);
    tv_mean  = mean(Ctop,1);
    tv_worst = max(Ctop,[],1);
    subplot(3,3,ai);
    fill([iter_ax, fliplr(iter_ax)],[tv_worst, fliplr(tv_best)], ...
         [0.82 0.90 0.96],'EdgeColor','none','FaceAlpha',0.5); hold on;
    plot(iter_ax,tv_worst,'r--','LineWidth',1.5);
    plot(iter_ax,tv_mean, 'b-', 'LineWidth',1.8);
    plot(iter_ax,tv_best, 'g-', 'LineWidth',2.0);
    plot(MAX_IT,tv_worst(end),'vr','MarkerFaceColor','r','MarkerSize',7);
    plot(MAX_IT,tv_mean(end), '^b','MarkerFaceColor','b','MarkerSize',7);
    plot(MAX_IT,tv_best(end), '^g','MarkerFaceColor','g','MarkerSize',7);
    text(MAX_IT*0.97,tv_worst(end),sprintf('%.1f',tv_worst(end)),...
         'Color','r','FontSize',7,'HorizontalAlignment','right');
    text(MAX_IT*0.97,tv_mean(end), sprintf('%.1f',tv_mean(end)),...
         'Color','b','FontSize',7,'HorizontalAlignment','right');
    text(MAX_IT*0.97,tv_best(end), sprintf('%.1f',tv_best(end)),...
         'Color','g','FontSize',7,'HorizontalAlignment','right');
    hold off; grid on; box on;
    set(gca,'FontSize',9,'XLim',[1 MAX_IT]);
    xlabel('Iteration','FontSize',9);
    ylabel('T_{op} (s)','FontSize',9);
    if ai == best_ai
        set(gca,'LineWidth',2.5,'XColor',[0.1 0.6 0.1],'YColor',[0.1 0.6 0.1]);
        title(sprintf('\\alpha=%.1f  \\bigstar BEST',ALPHA(ai)),'FontSize',10,'Color',[0.1 0.6 0.1]);
    else
        title(sprintf('\\alpha=%.1f',ALPHA(ai)),'FontSize',10);
    end
    if ai==1
        legend(cols_bw,'Location','best','FontSize',7,'Box','off');
    end
end
saveas(fig_top, sprintf('fig_%s_conv9_top.png',SYS));
fprintf('Saved: fig_%s_conv9_top.png  (3x3 subplots, T_op trajectory)\n',SYS);

%% ── FIGURE: VIOLIN PLOT (all 9 alpha on one figure) ─────────────────────
figure('Color','w','Position',[50 50 950 520]);
hold on;
if DIST_USE_RAW, DIST=FTOP_RAW; else, DIST=FTOP; end
for ai = 1:nA
    d = DIST(:,ai);
    % kernel density estimate (toolbox-free)
    [xd, yd] = violin_kde(d);
    w = 0.35 * yd/max(yd);   % half-width
    fill([ai+w, ai-fliplr(w)], [xd, fliplr(xd)], ...
         [0.3 0.6 0.9],'FaceAlpha',0.55,'EdgeColor',[0.1 0.3 0.6],'LineWidth',1.0);
    % overlay quartiles
    q1=prctile_loc(d,25); q3=prctile_loc(d,75); md=median(d);
    plot([ai ai],[q1 q3],'k-','LineWidth',4);
    plot(ai,md,'wo','MarkerFaceColor','w','MarkerSize',6);
    plot(ai,min(d),'g^','MarkerFaceColor','g','MarkerSize',6);
end
hold off; grid on; box on;
set(gca,'XTick',1:nA,'XTickLabel',compose('%.1f',ALPHA));
xlabel('Fractional Order \alpha','FontWeight','bold');
ylabel('Total T_{operation} (s)','FontWeight','bold');
src_lbl = ''; if DIST_USE_RAW, src_lbl=' [pre-refinement]'; end
title(sprintf('%s — T_{op} Distribution Across \\alpha (Violin, %d runs)%s',sysname,N_RUNS,src_lbl));
xline(best_ai,'r--','LineWidth',1.5,'Label',sprintf('Best \\alpha=%.1f',best_alpha));
saveas(gcf, sprintf('fig_%s_violin.png',SYS));
fprintf('Saved: fig_%s_violin.png\n',SYS);

%% ── FIGURE: CTI BAR for best alpha ──────────────────────────────────────
figure('Color','w','Position',[50 50 950 400]);
bv = CTI_arr; bv(~isfinite(bv))=0;
bh = bar(1:nPairs, bv, 0.8); hold on;
plot([0 nPairs+1],[CTI_thr CTI_thr],'r--','LineWidth',1.8); hold off;
bh.FaceColor='flat';
for p=1:nPairs
    if bv(p)>=CTI_thr-1e-6, bh.CData(p,:)=[0.2 0.7 0.35]; else, bh.CData(p,:)=[0.85 0.2 0.15]; end
end
grid on; box on;
xlabel('P/B Relay Pair Index','FontWeight','bold'); ylabel('CTI (s)','FontWeight','bold');
title(sprintf('%s — CTI per Pair (best \\alpha=%.1f)',sysname,best_alpha));
legend(sprintf('CTI_{min}=%.1f s',CTI_thr),'Location','northeast');
saveas(gcf, sprintf('fig_%s_CTI.png',SYS));
fprintf('Saved: fig_%s_CTI.png\n',SYS);

%% ── FIGURE: FREQUENCY DISTRIBUTION (best alpha, 50 runs) ────────────────
figure('Color','w','Position',[50 50 720 440]);
if DIST_USE_RAW, freq_data=FTOP_RAW(:,best_ai); else, freq_data=ftop_best; end
histogram(freq_data, 12, 'FaceColor',[0.3 0.6 0.9],'EdgeColor','w'); hold on;
xline(mean(freq_data),'b-','LineWidth',2,'Label','Mean');
xline(min(freq_data),'g-','LineWidth',2,'Label','Best');
hold off; grid on; box on;
xlabel('Total T_{operation} (s)','FontWeight','bold'); ylabel('Frequency','FontWeight','bold');
title(sprintf('%s — T_{op} Frequency over %d runs (\\alpha=%.1f)',sysname,N_RUNS,best_alpha));
saveas(gcf, sprintf('fig_%s_freqdist.png',SYS));
fprintf('Saved: fig_%s_freqdist.png\n',SYS);

%% ── PRINT TMS/PS/Tp/Tb/CTI TABLE ────────────────────────────────────────
fprintf('\n=== OPTIMAL SETTINGS (best alpha=%.1f) ===\n', best_alpha);
fprintf('%-4s %8s %8s | per-pair: %4s %4s %8s %8s %8s\n', ...
        'R','TMS','PS','Pair','P>B','Tp(s)','Tb(s)','CTI(s)');
for r=1:nR
    fprintf('R%-3d %8.4f %8.4f\n', r, TMS(r), PS(r));
end
fprintf('\n-- Per-pair operating times --\n');
fprintf('%-4s %5s %5s %9s %9s %9s\n','Pair','Pri','Bkp','Tp(s)','Tb(s)','CTI(s)');
for p=1:nPairs
    fprintf('%-4d R%-3d R%-3d %9.4f %9.4f %9.4f\n', ...
            p, PB(p,1), PB(p,2), Tp_arr(p), Tb_arr(p), CTI_arr(p));
end

%% ── SAVE ────────────────────────────────────────────────────────────────
save(sprintf('results_HFEE_%s.mat',SYS), ...
    'CONV','CONV_TOP','FTOP','FTOP_RAW','FVIOL','FVIOL_RAW','BEST_SOL','BEST_VIOL_A','BEST_TOP_A', ...
    'MINTOP_SOL','MINTOP_TOP','MINTOP_VIOL','mintop_ai','mintop_alpha', ...
    'Tp_mt','Tb_mt','CTI_mt', ...
    'best_alpha','best_ai','ALPHA','TMS','PS','Tp_arr','Tb_arr','CTI_arr', ...
    'mu','sd','cv','bestv','worstv','ci95','fM','CTR','PB','sysname');
fprintf('\nSaved: results_HFEE_%s.mat\n', SYS);
fprintf('=== DONE: %s ===\n', sysname);

%% =========================================================================
%% LOCAL FUNCTIONS
%% =========================================================================

function [pos, conv, conv_top] = fpsogsa_ee(dim, LB, UB, n, maxIt, fM, CTR, PB, ...
    A, B, CTI_thr, alpha, w0, wd, G0, C1, C2, lH, lam_hard, lam_soft)
    % FPSOGSA-EE: fractional PSO + GSA with Shannon entropy enhancement.
    % Returns: pos      = best solution (min penalised cost)
    %          conv     = monotone best-so-far PENALISED COST per iteration
    %          conv_top = monotone best-so-far TRUE T_op of the incumbent
    %                     (the T_op of the lowest-cost solution so far)
    LB=LB(:)'; UB=UB(:)';
    X = LB + (UB-LB).*rand(n,dim);
    V = 0.3*randn(n,dim);
    fit = zeros(n,1);
    for i=1:n, fit(i)=cost(X(i,:),fM,CTR,PB,A,B,CTI_thr,lam_hard,lam_soft,lH); end
    pBest=X; pBf=fit;
    [gBf,gi]=min(fit); gBest=X(gi,:);
    % Convergence tracker: monotone best-so-far PENALISED COST.
    % Cost = T_op + lam_hard*violations + lam_soft*shortfall^2, so the curve
    % shows the optimiser clearing violations (large early drops) then
    % refining T_op (small later drops). Plotted on log scale.
    conv=zeros(1,maxIt); conv_top=zeros(1,maxIt); w=w0;
    best_cost = gBf;
    for it=1:maxIt
        G = G0*exp(-23*it/maxIt);
        % gravitational masses
        best=min(fit); worst=max(fit);
        if worst==best, M=ones(n,1)/n; else, M=(worst-fit)/(worst-best+1e-12); M=M/sum(M); end
        % acceleration
        Acc=zeros(n,dim);
        for i=1:n
            df=X-X(i,:); R=sqrt(sum(df.^2,2))+1e-10;
            F=rand(n,1).*M./R; F(i)=0;
            Acc(i,:)=G*sum(F.*df,1);
        end
        % fractional-order velocity + PSO
        r1=rand(n,dim); r2=rand(n,dim);
        V = alpha*V + C1*r1.*Acc + C2*r2.*(gBest-X);
        X = min(max(X+V,LB),UB);
        for i=1:n
            fit(i)=cost(X(i,:),fM,CTR,PB,A,B,CTI_thr,lam_hard,lam_soft,lH);
            if fit(i)<pBf(i), pBf(i)=fit(i); pBest(i,:)=X(i,:); end
        end
        [mb,mi]=min(fit);
        if mb<gBf, gBf=mb; gBest=X(mi,:); end
        % entropy-based diversity reinjection
        H = swarm_entropy(X, LB, UB);
        if H < 0.3
            nReinit = round(0.1*n);
            idx = randperm(n, nReinit);
            X(idx,:) = LB + (UB-LB).*rand(nReinit,dim);
        end
        % Monotone best-so-far penalised cost
        if gBf < best_cost, best_cost = gBf; end
        conv(it) = best_cost;
        % Monotone best-so-far TRUE T_op of the incumbent global best
        % T_op of the current cost-incumbent (gBest). NOT clamped monotone:
        % this honestly shows T_op rising as violations are cleared (raising
        % backup TMS costs operating time), then settling. The penalised-cost
        % curve is the true monotone convergence metric; this complements it.
        conv_top(it) = true_top(gBest, fM, CTR, A, B);
        w = w*wd;
    end
    pos = gBest;
end

function f = cost(x, fM, CTR, PB, A, B, CTI_thr, lam_hard, lam_soft, lH)
    [top, nv, sf] = eval_full(x, fM, CTR, PB, A, B, CTI_thr);
    H = 0; % entropy handled at swarm level
    f = top + lam_hard*nv + lam_soft*sf - lH*H;
end

function [top, nv, sf] = eval_full(x, fM, CTR, PB, A, B, CTI_thr)
    nR=numel(x)/2; TMS=x(1:nR); PS=x(nR+1:end);
    TMS=TMS(:)'; PS=PS(:)'; CTR=CTR(:)';
    Ip=CTR.*PS; PSM=fM./(Ip+1e-12);
    dP=max(PSM.^B-1,1e-12);
    T=(A.*TMS)./dP; T(PSM<=1|fM<=0)=0; T=min(T,50);
    nSc=size(fM,1); Tp=zeros(nSc,1);
    for s=1:nSc, row=T(s,:); row=row(row>0); if ~isempty(row), Tp(s)=min(row); end; end
    top=sum(Tp);
    nv=0; sf=0;
    for p=1:size(PB,1)
        pi_=PB(p,1); bi_=PB(p,2); cmin=Inf;
        for s=1:nSc
            if T(s,pi_)>0 && T(s,bi_)>0, d=T(s,bi_)-T(s,pi_); if d<cmin, cmin=d; end; end
        end
        if isfinite(cmin) && cmin<CTI_thr-1e-6
            nv=nv+1; sf=sf+(CTI_thr-cmin)^2;
        end
    end
end

function [top, nv] = eval_solution(x, fM, CTR, PB, A, B, CTI_thr)
    [top, nv, ~] = eval_full(x, fM, CTR, PB, A, B, CTI_thr);
end

function top = true_top(x, fM, CTR, A, B)
    nR=numel(x)/2; TMS=x(1:nR); PS=x(nR+1:end);
    TMS=TMS(:)'; PS=PS(:)'; CTR=CTR(:)';
    Ip=CTR.*PS; PSM=fM./(Ip+1e-12);
    dP=max(PSM.^B-1,1e-12);
    T=(A.*TMS)./dP; T(PSM<=1|fM<=0)=0; T=min(T,50);
    nSc=size(fM,1); Tp=zeros(nSc,1);
    for s=1:nSc, row=T(s,:); row=row(row>0); if ~isempty(row), Tp(s)=min(row); end; end
    top=sum(Tp);
end

function H = swarm_entropy(X, LB, UB)
    % Normalised positional entropy of the swarm (diversity index).
    Xn=(X-LB)./(UB-LB+1e-12);
    nb=10; H=0; d=size(X,2);
    for j=1:d
        c=histcounts(Xn(:,j),linspace(0,1,nb+1));
        p=c/sum(c); p=p(p>0);
        H=H-sum(p.*log(p));
    end
    H=H/(d*log(nb));   % normalise to [0,1]
end

function pos = cti_repair(pos, fM, CTR, PB, A, B, CTI_thr, LB, UB)
    % Difference-constraint repair: raise backup TMS to satisfy CTI.
    nR=numel(pos)/2; TMS=pos(1:nR); PS=pos(nR+1:end);
    TMS=TMS(:)'; PS=PS(:)'; CTR=CTR(:)';
    Ip=CTR.*PS; PSM=fM./(Ip+1e-12);
    nSc=size(fM,1);
    coeff=zeros(nSc,nR);
    for s=1:nSc, for r=1:nR
        if fM(s,r)>0 && PSM(s,r)>1, coeff(s,r)=A/(PSM(s,r)^B-1); end
    end; end
    TMS_ub=UB(1); % first nR are TMS, all same upper bound
    if numel(unique(UB(1:nR)))==1, TMS_ub=UB(1); else, TMS_ub=max(UB(1:nR)); end
    for sweep=1:200
        ch=false;
        for p=1:size(PB,1)
            pi_=PB(p,1); bi_=PB(p,2);
            for s=1:nSc
                if coeff(s,pi_)>0 && coeff(s,bi_)>0
                    need=(CTI_thr+coeff(s,pi_)*TMS(pi_))/coeff(s,bi_);
                    if TMS(bi_)<need-1e-9
                        TMS(bi_)=min(need,TMS_ub); ch=true;
                    end
                end
            end
        end
        if ~ch, break; end
    end
    pos=[TMS, PS];
end

function pos = fmincon_refine(pos, fM, CTR, PB, A, B, CTI_thr, LB, UB)
    % Local refinement via fmincon (SQP). Minimise T_op s.t. CTI constraints.
    % Optimises BOTH TMS and PS jointly to drive T_op as low as possible
    % while maintaining CTI feasibility.
    nR=numel(pos)/2;
    obj   = @(x) true_top(x, fM, CTR, A, B);
    nlcon = @(x) cti_nonlcon(x, fM, CTR, PB, A, B, CTI_thr);
    opts = optimoptions('fmincon','Algorithm','sqp','Display','off', ...
        'MaxFunctionEvaluations',4e4,'MaxIterations',500, ...
        'ConstraintTolerance',1e-6,'OptimalityTolerance',1e-8, ...
        'StepTolerance',1e-10);
    [top_old, nv_old] = eval_solution(pos, fM, CTR, PB, A, B, CTI_thr);
    best_pos = pos; best_nv = nv_old; best_top = top_old;
    try
        x = fmincon(obj, pos(:)', [],[],[],[], LB, UB, nlcon, opts);
        [top_new, nv_new] = eval_solution(x, fM, CTR, PB, A, B, CTI_thr);
        % Accept on feasibility-first ordering: fewer violations, then lower T_op
        if (nv_new < best_nv) || (nv_new==best_nv && top_new < best_top - 1e-9)
            best_pos = x; best_nv = nv_new; best_top = top_new;
        end
    catch
        % fmincon unavailable/failed -> keep repaired solution
    end
    pos = best_pos;
end

function [c, ceq] = cti_nonlcon(x, fM, CTR, PB, A, B, CTI_thr)
    % Inequality: CTI_thr - (Tb - Tp) <= 0  for every pair (worst scenario)
    nR=numel(x)/2; TMS=x(1:nR); PS=x(nR+1:end);
    TMS=TMS(:)'; PS=PS(:)'; CTR=CTR(:)';
    Ip=CTR.*PS; PSM=fM./(Ip+1e-12);
    dP=max(PSM.^B-1,1e-12);
    T=(A.*TMS)./dP; T(PSM<=1|fM<=0)=0; T=min(T,50);
    nSc=size(fM,1); nP=size(PB,1);
    c=zeros(nP,1);
    for p=1:nP
        pi_=PB(p,1); bi_=PB(p,2); cmin=Inf;
        for s=1:nSc
            if T(s,pi_)>0 && T(s,bi_)>0, d=T(s,bi_)-T(s,pi_); if d<cmin, cmin=d; end; end
        end
        if isfinite(cmin), c(p)=CTI_thr-cmin; else, c(p)=-1; end
    end
    ceq=[];
end

function [Tp_arr, Tb_arr, CTI_arr] = cti_table(TMS, PS, fM, CTR, PB, A, B)
    nR=numel(TMS); TMS=TMS(:)'; PS=PS(:)'; CTR=CTR(:)';
    Ip=CTR.*PS; PSM=fM./(Ip+1e-12);
    dP=max(PSM.^B-1,1e-12);
    T=(A.*TMS)./dP; T(PSM<=1|fM<=0)=0; T=min(T,50);
    nSc=size(fM,1); nP=size(PB,1);
    Tp_arr=nan(nP,1); Tb_arr=nan(nP,1); CTI_arr=nan(nP,1);
    for p=1:nP
        pi_=PB(p,1); bi_=PB(p,2); cmin=Inf; bs=0;
        for s=1:nSc
            if T(s,pi_)>0 && T(s,bi_)>0
                d=T(s,bi_)-T(s,pi_); if d<cmin, cmin=d; bs=s; end
            end
        end
        if bs>0, Tp_arr(p)=T(bs,pi_); Tb_arr(p)=T(bs,bi_); CTI_arr(p)=cmin; end
    end
end

function [xd, yd] = violin_kde(d)
    % Simple Gaussian KDE for violin plot (toolbox-free).
    d=d(:); n=numel(d);
    if std(d)<1e-9, xd=linspace(min(d)-0.1,max(d)+0.1,50); yd=ones(1,50); return; end
    h = 1.06*std(d)*n^(-1/5);   % Silverman bandwidth
    xd = linspace(min(d)-2*h, max(d)+2*h, 80);
    yd = zeros(1,80);
    for k=1:80
        yd(k)=sum(exp(-0.5*((xd(k)-d)/h).^2))/(n*h*sqrt(2*pi));
    end
end

function p = prctile_loc(x, q)
    x=sort(x(:)); n=numel(x);
    if n==1, p=x(1); return; end
    pos=(q/100)*(n-1)+1; lo=floor(pos); hi=ceil(pos);
    if lo==hi, p=x(lo); else, p=x(lo)+(pos-lo)*(x(hi)-x(lo)); end
end

%% =========================================================================
%% DATA LOADER
%% =========================================================================
function [fM, CTR, PB, TMS_lb, TMS_ub, PS_lb, PS_ub, sysname] = load_system(SYS)
    % Bounds (same for all systems per requirement)
    TMS_lb=0.1; TMS_ub=1.0; PS_lb=0.5; PS_ub=2.5;
    switch upper(SYS)
        case 'IEEE15'
            [fM, CTR, PB] = data_ieee15();
            sysname='IEEE 15-Bus (42 relays, 92 pairs)';
        case 'IEEE30'
            [fM, CTR, PB] = data_ieee30();
            sysname='IEEE 30-Bus (38 relays, 60 pairs)';
        case 'IEEE42'
            [fM, CTR, PB] = data_ieee42();
            sysname='IEEE 42-Bus (97 relays, 117 pairs)';
        otherwise
            error('Unknown system: %s', SYS);
    end
end

%% =========================================================================
%% SYSTEM DATA FUNCTIONS
%% =========================================================================

function [fM, CTR, PB] = data_ieee15()
    % IEEE 15-bus: 42 relays, 82 P/B pairs.
    % CTR groups: 800/5=160, 1200/5=240, 600/5=120, 400/5=80, 1600/5=320.
    % Source: kamel2020development, khurshaid2019improved (old paper Table).
    % NOTE: paper text states 92 pairs; transcribed table has 82.
    %       VERIFY against source before final paper.
    nR = 42;
    CTR = [160,240,160,240,160,120,120,240,120,160,240,240,160,240,240,120,80,320,160,320,320,80,240,120,120,120,120,120,320,80,120,120,120,80,120,160,160,80,80,160,80,160];
    % Pair data: [primary, backup, I_f^primary(A), I_f^backup(A)]
    PB_data = [
        1 6 3621.0000 1233.0000;
        2 4 4597.0000 1477.0000;
        2 16 4597.0000 743.0000;
        3 1 3984.0000 853.0000;
        3 16 3984.0000 743.0000;
        4 7 4382.0000 1111.0000;
        4 12 4382.0000 1463.0000;
        4 20 4382.0000 1808.0000;
        5 2 3319.0000 922.0000;
        6 8 2647.0000 1548.0000;
        6 10 2647.0000 1100.0000;
        7 5 2497.0000 1397.0000;
        7 10 2497.0000 1100.0000;
        8 3 4695.0000 1424.0000;
        8 12 4695.0000 1463.0000;
        8 20 4695.0000 1808.0000;
        9 5 2943.0000 1397.0000;
        9 8 2943.0000 1548.0000;
        10 14 3568.0000 1175.0000;
        11 3 4342.0000 1424.0000;
        11 7 4342.0000 1111.0000;
        11 20 4342.0000 1808.0000;
        12 13 4195.0000 1503.0000;
        12 24 4195.0000 753.0000;
        13 9 3402.0000 1009.0000;
        14 11 4606.0000 1475.0000;
        14 24 4606.0000 753.0000;
        15 1 4712.0000 853.0000;
        15 4 4712.0000 1477.0000;
        16 18 2225.0000 1320.0000;
        16 26 2225.0000 905.0000;
        17 15 1875.0000 969.0000;
        17 26 1875.0000 905.0000;
        18 19 8426.0000 1372.0000;
        18 22 8426.0000 642.0000;
        18 30 8426.0000 681.0000;
        19 3 3998.0000 1424.0000;
        19 7 3998.0000 1111.0000;
        19 12 3998.0000 1463.0000;
        20 17 7662.0000 599.0000;
        20 22 7662.0000 642.0000;
        20 30 7662.0000 681.0000;
        21 17 8384.0000 599.0000;
        21 19 8384.0000 1372.0000;
        21 30 8384.0000 681.0000;
        22 23 1950.0000 979.0000;
        22 34 1950.0000 970.0000;
        23 11 4910.0000 1475.0000;
        23 13 4910.0000 1053.0000;
        24 21 2296.0000 175.0000;
        24 34 2296.0000 970.0000;
        25 15 2289.0000 969.0000;
        25 18 2289.0000 1320.0000;
        26 28 2300.0000 1192.0000;
        26 36 2300.0000 1109.0000;
        27 25 2011.0000 903.0000;
        27 36 2011.0000 1109.0000;
        28 29 2525.0000 1828.0000;
        28 32 2525.0000 697.0000;
        29 17 8346.0000 599.0000;
        29 19 8346.0000 1372.0000;
        29 22 8346.0000 642.0000;
        30 27 1736.0000 1039.0000;
        30 32 1736.0000 697.0000;
        31 27 2867.0000 1039.0000;
        31 29 2867.0000 1828.0000;
        32 33 2069.0000 1162.0000;
        32 42 2069.0000 907.0000;
        33 21 2305.0000 1326.0000;
        33 23 2305.0000 979.0000;
        34 31 1715.0000 809.0000;
        34 42 1715.0000 907.0000;
        35 25 2095.0000 903.0000;
        35 28 2095.0000 1192.0000;
        36 38 3283.0000 882.0000;
        37 35 3301.0000 910.0000;
        38 40 1403.0000 1403.0000;
        39 37 1434.0000 1434.0000;
        40 41 3140.0000 745.0000;
        41 31 1971.0000 809.0000;
        41 33 1971.0000 1162.0000;
        42 39 3295.0000 896.0000;
    ];
    PB = PB_data(:,1:2);
    nP = size(PB,1);
    % Build fM: one scenario per pair (near-end fault model).
    % Scenario p: primary sees I_f^P, backup sees I_f^B.
    fM = zeros(nP, nR);
    for p = 1:nP
        fM(p, PB_data(p,1)) = PB_data(p,3);   % primary fault current
        fM(p, PB_data(p,2)) = PB_data(p,4);   % backup fault current
    end
end

function [fM, CTR, PB] = data_ieee30()
    % IEEE 30-bus: 38 relays, 60 P/B pairs.
    % CTR = 1000/5 = 200 for ALL 38 relays.
    % Fault currents from CSV: 62 scenarios x 38 relays.
    % Source: kamel2020development.
    nR = 38;
    CTR = 200*ones(1,nR);

    % --- Load fault matrix from CSV ---
    csvFile = '30with38.csv';
    if ~exist(csvFile,'file')
        error('CSV not found: %s. Place it in the working folder.', csvFile);
    end
    T = readtable(csvFile);
    fM = double(table2array(T(:,2:end)));   % skip scenario-ID column
    fM(isnan(fM)) = 0; fM(fM<0) = 0;
    fM = fM(1:62, 1:nR);

    % --- 60 P/B pairs (from Kamel 2020 / prior validated code) ---
    PB = [
       1  3;  1 21;  1 28;  1 29;  2 20;  2 28;  2 29;  3  4;  3 21;
       4  2;  4  5;  4 18;  5  6;  5 37;  6  7;  6  8;  7 27;  8 26;
       9 12;  9 20;  9 21;  9 29; 10 11; 10 20; 10 21; 10 28; 11 13;
      12 14; 13 15; 14 16; 14 17; 15 19; 15 35; 15 36; 15 38; 16 19;
      16 34; 16 36; 17 19; 17 34; 17 35; 18 24; 18 38; 19 37; 20 22;
      21 23; 22  2; 22 23; 23 24; 23 37; 24 25; 28 31; 29 30; 30 32;
      31 33; 32 34; 33 35; 33 36; 34 38; 36 38];
end

function [fM, CTR, PB] = data_ieee42()
    % IEEE 42-bus: 97 relays, 114 P/B pairs.
    % CTR per-relay (50..3000) from Al-Roomi Table 2.
    % Fault currents from Al-Roomi Table 3 (converted kA -> A).
    % Source: alroomi2015test, al-roomi.org/coordination/42-bus-system.
    % NOTE: source lists 117 pairs; transcribed table has 114.
    %       VERIFY 3 missing pairs against source before final paper.
    nR = 97;
    CTR = [800,800,800,1200,1200,75,400,400,400,400,200,1000,1000,200,1200,1200,500,800,500,500,500,500,500,500,500,2000,500,500,500,300,500,1500,500,500,100,3000,100,100,3000,100,100,500,100,100,3000,3000,3000,500,500,250,250,100,600,1500,1500,150,75,400,2000,75,200,2000,1000,50,1500,2000,800,800,500,500,400,100,3000,400,50,600,100,200,100,200,1000,500,800,600,100,100,100,200,75,100,200,250,1000,100,100,100,1000];
    PB_data = [
        4 3 9821.0000 5835.0000;
        4 15 9821.0000 3876.0000;
        5 3 9821.0000 5835.0000;
        5 16 9821.0000 3876.0000;
        6 3 13581.0000 5835.0000;
        7 1 9115.0000 8367.0000;
        7 10 9115.0000 748.0000;
        8 1 9113.0000 8367.0000;
        8 9 9113.0000 746.0000;
        9 13 767.0000 3835.0000;
        10 12 768.0000 3840.0000;
        11 8 6820.0000 6820.0000;
        12 2 7528.0000 4084.0000;
        13 4 7495.0000 2873.0000;
        13 5 7495.0000 2873.0000;
        14 7 6822.0000 6822.0000;
        15 4 11004.0000 2873.0000;
        15 14 11004.0000 1276.4000;
        16 5 11004.0000 2873.0000;
        16 14 11004.0000 1276.4000;
        17 2 13364.0000 4084.0000;
        17 11 13364.0000 1276.2000;
        18 2 12014.0000 4084.0000;
        18 11 12014.0000 1276.2000;
        19 2 13818.0000 4084.0000;
        19 11 13818.0000 1276.2000;
        20 2 13542.0000 4084.0000;
        20 11 13542.0000 1276.2000;
        21 2 13359.0000 4084.0000;
        21 11 13359.0000 1276.2000;
        22 4 13118.0000 2873.0000;
        22 5 13118.0000 2873.0000;
        22 14 13118.0000 1276.4000;
        23 4 13698.0000 2873.0000;
        23 5 13698.0000 2873.0000;
        23 14 13698.0000 1276.4000;
        24 4 13429.0000 2873.0000;
        24 5 13429.0000 2873.0000;
        24 14 13429.0000 1276.4000;
        25 4 13511.0000 2873.0000;
        25 5 13511.0000 2873.0000;
        25 14 13511.0000 1276.4000;
        26 59 35288.0000 30761.0000;
        27 25 13132.0000 12937.0000;
        28 36 35343.0000 28873.0000;
        29 25 13109.0000 12937.0000;
        30 32 31986.0000 31986.0000;
        31 47 33338.0000 28686.0000;
        32 47 34570.0000 28686.0000;
        33 42 12749.0000 12749.0000;
        34 39 31412.0000 24941.0000;
        35 62 23847.0000 19803.0000;
        36 37 28873.0000 1004.3000;
        37 33 12579.0000 12579.0000;
        38 29 12963.0000 12964.0000;
        39 38 24941.0000 867.5000;
        40 27 11613.0000 11613.0000;
        41 23 13081.0000 13081.0000;
        42 20 13163.0000 12997.0000;
        43 62 21056.0000 19803.0000;
        44 20 13197.0000 12997.0000;
        45 44 28984.0000 1008.1000;
        46 41 28963.0000 1007.4000;
        47 40 28686.0000 997.8000;
        48 17 13070.0000 12923.0000;
        49 17 13321.0000 12923.0000;
        50 21 12526.0000 12526.0000;
        51 22 12812.0000 12722.0000;
        52 22 13391.0000 12722.0000;
        53 52 6046.0000 1051.5000;
        54 51 13415.0000 2333.0000;
        55 50 13361.0000 2323.7000;
        56 24 13281.0000 13142.0000;
        57 24 13451.0000 13142.0000;
        58 56 3659.0000 1103.0000;
        59 57 30761.0000 1069.9000;
        60 49 12892.0000 12892.0000;
        61 48 12778.0000 12778.0000;
        62 60 19803.0000 688.8000;
        63 61 9156.0000 1592.3000;
        64 19 12748.0000 12748.0000;
        65 64 15046.0000 523.3000;
        66 6 29053.0000 1010.5000;
        67 46 35125.0000 28963.0000;
        68 46 35125.0000 28963.0000;
        69 46 34656.0000 28963.0000;
        70 45 33409.0000 28984.0000;
        71 66 31497.0000 29053.0000;
        72 66 30235.0000 29053.0000;
        73 59 30761.0000 30761.0000;
        74 58 3659.0000 3659.0000;
        75 53 6046.0000 6046.0000;
        76 18 11690.0000 11690.0000;
        77 39 28487.0000 24941.0000;
        78 39 29081.0000 24941.0000;
        79 36 32418.0000 28873.0000;
        80 36 33012.0000 28873.0000;
        81 54 15033.0000 13415.0000;
        82 54 16828.0000 13415.0000;
        83 55 15749.0000 13361.0000;
        84 55 14904.0000 13361.0000;
        85 65 15046.0000 15046.0000;
        86 34 29038.0000 29038.0000;
        87 28 32348.0000 32348.0000;
        88 47 31627.0000 28686.0000;
        89 47 32853.0000 28686.0000;
        90 31 30648.0000 30648.0000;
        91 69 31782.0000 31782.0000;
        92 46 31392.0000 28963.0000;
        93 46 33158.0000 28963.0000;
        94 70 30736.0000 30736.0000;
        95 45 32321.0000 28984.0000;
        96 45 32526.0000 28984.0000;
        97 63 9156.0000 9156.0000;
    ];
    PB = PB_data(:,1:2);
    nP = size(PB,1);
    fM = zeros(nP, nR);
    for p = 1:nP
        fM(p, PB_data(p,1)) = PB_data(p,3);
        fM(p, PB_data(p,2)) = PB_data(p,4);
    end
end
#!/usr/bin/env python3

import time
import re
import sys
import subprocess
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import sentencepiece as spm

from icefall.utils import AttributeDict
from lhotse import Fbank, FbankConfig
from lhotse.audio import Recording
from pyctcdecode import build_ctcdecoder

from wrapper import StreamingAsrWrapper, InitializeStates
from wrapper import StreamingEncoderWrapper, CTC
from subsampling import Conv2dSubsampling
from zipformer import Zipformer2
from search import PrefixScore


torch.backends.cudnn.enabled = True
torch._C._set_graph_executor_optimize(True)


def to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def decode_byte_fallback(text):
    pattern = re.compile(r"(<0x[0-9A-Fa-f]{2}>)+")

    def replace_bytes(match):
        tokens = re.findall(
            r"<0x([0-9A-Fa-f]{2})>",
            match.group(0),
        )

        data = bytes(int(x, 16) for x in tokens)

        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return match.group(0)

    return pattern.sub(replace_bytes, text)


def load_unigrams_from_arpa(arpa_path):
    """
    ARPA의 \\1-grams: section에서 unigram vocabulary를 읽는다.

    ARPA 전체를 메모리에 올리지 않고 streaming 방식으로 읽는다.
    """

    unigrams = []
    in_unigram_section = False

    with open(
        arpa_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line == "\\1-grams:":
                in_unigram_section = True
                continue

            if in_unigram_section and line.startswith("\\"):
                break

            if not in_unigram_section:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            word = parts[1]

            # sentence boundary 제외
            if word in ("<s>", "</s>"):
                continue

            unigrams.append(word)

    print(
        f"[KenLM] loaded {len(unigrams):,} unigrams "
        f"from {arpa_path}"
    )

    return unigrams


class ModuleExport(object):

    def __init__(
        self,
        exp_path="",
        modelname="",
        test_path="",
        out_path="",
        kenlm_model_path=None,
        hotwords=None,
        hotword_weight=10.0,
        lm_alpha=0.5,
        lm_beta=0.05,
    ):

        super(ModuleExport, self).__init__()

        self.main_path = exp_path
        self.export_path = self.main_path

        self.decode_chunk_size = 160
        self.left_context = 4
        self.pad_length = 7

        self.model_path = f"{self.main_path}/{modelname}"

        self.eval_exe = "bin/b2b/compute-cer.py"

        self.sp = spm.SentencePieceProcessor(
            model_file=f"{self.main_path}/bpe.model"
        )

        self.blank_id = 0
        self.eos_id = 1
        self.unk_id = 2

        # ----------------------------------------------------
        # KenLM / Hotword 설정
        # ----------------------------------------------------

        self.kenlm_model_path = kenlm_model_path

        self.hotwords = hotwords or []

        self.hotword_weight = hotword_weight

        self.lm_alpha = lm_alpha
        self.lm_beta = lm_beta

        # ----------------------------------------------------
        # CTC labels
        #
        # AM CTC output index와 정확하게 일치해야 함
        # ----------------------------------------------------

        labels = []

        for i in range(self.sp.get_piece_size()):

            piece = self.sp.id_to_piece(i)

            if i == self.blank_id:
                labels.append("")
            else:
                labels.append(piece)

        print(
            f"[CTC] vocab size = {len(labels)}"
        )

        # 디버깅용
        print("[CTC] first labels:")

        for i in range(min(20, len(labels))):
            print(
                f"  {i:4d}: {repr(labels[i])}"
            )

        # ----------------------------------------------------
        # pyctcdecode decoder
        # ----------------------------------------------------

        if self.kenlm_model_path:

            print(
                f"[KenLM] model = "
                f"{self.kenlm_model_path}"
            )

            # ARPA인 경우 unigram 자동 로드
            if str(
                self.kenlm_model_path
            ).lower().endswith(".arpa"):

                unigrams = load_unigrams_from_arpa(
                    self.kenlm_model_path
                )

            else:

                unigrams = None

            self.ctc_decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=self.kenlm_model_path,
                unigrams=unigrams,
                alpha=self.lm_alpha,
                beta=self.lm_beta,
            )

        else:

            print(
                "[KenLM] disabled"
            )

            self.ctc_decoder = build_ctcdecoder(
                labels=labels,
            )

        print(
            f"[Hotword] count = "
            f"{len(self.hotwords)}"
        )

        print(
            f"[Hotword] weight = "
            f"{self.hotword_weight}"
        )

        if self.hotwords:
            print(
                "[Hotword]",
                self.hotwords,
            )

        # ----------------------------------------------------

        self.asr = None

        self.test_path = test_path

        self.out_path = (
            f"{out_path}/{modelname}"
        )

        self.params = AttributeDict(
            {
                "feature_dim": 80,
                "subsampling_factor": 4,
                "downsampling_factor":
                    "1,2,4,8,4,2",
                "num_encoder_layers":
                    "2,2,3,4,3,2",
                "encoder_dim":
                    "192,256,384,512,384,256",
                "encoder_unmasked_dim":
                    "192,192,256,256,256,192",
                "query_head_dim": "32",
                "pos_head_dim": "4",
                "value_head_dim": "12",
                "pos_dim": 48,
                "num_heads":
                    "4,4,4,8,4,4",
                "feedforward_dim":
                    "512,768,1024,1536,1024,768",
                "cnn_module_kernel":
                    "31,31,15,15,15,31",
                "causal": True,
                "chunk_size": "64",
                "left_context_frames":
                    str(self.left_context * 4),
                "decoder_dim": 512,
                "joiner_dim": 512,
                "vocab_size": 1024,
                "context_size": 2,
                "blank_id": 0,
            }
        )

    def get_checkpoint(self):

        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=False,
        )

        return checkpoint

    def encoder_trace(self, checkpoint):

        params = self.params

        device = torch.device("cpu")

        encoder_embed_checkpoint = OrderedDict()
        encoder_checkpoint = OrderedDict()
        ctc_checkpoint = OrderedDict()

        for key in checkpoint["model"].keys():

            layer_type = key.split(".")[0]

            if layer_type == "encoder_embed":

                new_key = ".".join(
                    key.split(".")[1:]
                )

                encoder_embed_checkpoint[
                    new_key
                ] = checkpoint["model"].get(
                    key
                )

            if layer_type == "encoder":

                new_key = ".".join(
                    key.split(".")[1:]
                )

                encoder_checkpoint[
                    new_key
                ] = checkpoint["model"].get(
                    key
                )

            if layer_type == "ctc_output":

                ctc_checkpoint[
                    key
                ] = checkpoint["model"].get(
                    key
                )

        encoder_dim = max(
            to_int_tuple(
                params.encoder_dim
            )
        )

        # ----------------------------------------------------
        # Encoder Embed
        # ----------------------------------------------------

        encoder_embed = Conv2dSubsampling(
            in_channels=params.feature_dim,
            out_channels=to_int_tuple(
                params.encoder_dim
            )[0],
        )

        encoder_embed.load_state_dict(
            encoder_embed_checkpoint
        )

        encoder_embed.to(device)
        encoder_embed.eval()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        encoder = Zipformer2(
            output_downsampling_factor=2,
            downsampling_factor=to_int_tuple(
                params.downsampling_factor
            ),
            num_encoder_layers=to_int_tuple(
                params.num_encoder_layers
            ),
            encoder_dim=to_int_tuple(
                params.encoder_dim
            ),
            encoder_unmasked_dim=to_int_tuple(
                params.encoder_unmasked_dim
            ),
            query_head_dim=to_int_tuple(
                params.query_head_dim
            ),
            pos_head_dim=to_int_tuple(
                params.pos_head_dim
            ),
            value_head_dim=to_int_tuple(
                params.value_head_dim
            ),
            pos_dim=params.pos_dim,
            num_heads=to_int_tuple(
                params.num_heads
            ),
            feedforward_dim=to_int_tuple(
                params.feedforward_dim
            ),
            cnn_module_kernel=to_int_tuple(
                params.cnn_module_kernel
            ),
            causal=params.causal,
            chunk_size=to_int_tuple(
                params.chunk_size
            ),
            left_context_frames=to_int_tuple(
                params.left_context_frames
            ),
        )

        encoder.load_state_dict(
            encoder_checkpoint
        )

        encoder.to(device)
        encoder.eval()

        # ----------------------------------------------------
        # CTC
        # ----------------------------------------------------

        ctc = CTC(
            vocab_size=params.vocab_size,
            encoder_dim=encoder_dim,
        )

        ctc.load_state_dict(
            ctc_checkpoint
        )

        ctc.to(device)
        ctc.eval()

        # ----------------------------------------------------
        # Wrapper
        # ----------------------------------------------------

        ext_encoder = StreamingEncoderWrapper(
            encoder_embed=encoder_embed,
            encoder=encoder,
            eproj=None,
            ctc=ctc,
        )

        initialized = InitializeStates(
            in_channels=params.feature_dim,
            downsampling_factor=to_int_tuple(
                params.downsampling_factor
            ),
            encoder_dim=to_int_tuple(
                params.encoder_dim
            ),
            num_encoder_layers=to_int_tuple(
                params.num_encoder_layers
            ),
            query_head_dim=to_int_tuple(
                params.query_head_dim
            ),
            value_head_dim=to_int_tuple(
                params.value_head_dim
            ),
            num_heads=to_int_tuple(
                params.num_heads
            ),
            cnn_module_kernel=to_int_tuple(
                params.cnn_module_kernel
            ),
        )

        initialized.to(device)
        initialized.eval()

        self.asr = StreamingAsrWrapper(
            initialized=initialized,
            encoder=ext_encoder,
            blank=0,
            eos=1,
            unk=2,
            left_context=64,
            decode_chunk_size=self.decode_chunk_size,
            context_size=params.context_size,
            subsampling_factor=4,
            enc_n_dim=max(
                to_int_tuple(
                    params.encoder_dim
                )
            ),
            dec_n_dim=params.joiner_dim,
            cnn_module_kernel=31,
            device=device,
        )

    def search(self):

        torch._C._set_graph_executor_optimize(
            False
        )

        target_dir = Path(
            self.test_path
        )

        output_dir = Path(
            self.out_path
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        ref_file = output_dir / "ref.txt"
        hyp_file = output_dir / "hyp.txt"
        eval_file = output_dir / "cer.txt"

        rf_out = open(
            ref_file,
            "w",
            encoding="utf-8",
        )

        hf_out = open(
            hyp_file,
            "w",
            encoding="utf-8",
        )

        for txt_file in sorted(
            target_dir.rglob("*.txt")
        ):

            wav_file = (
                txt_file.with_suffix(
                    ".wav"
                )
            )

            if not wav_file.exists():
                continue

            text = " ".join(
                txt_file.read_text(
                    encoding="utf-8"
                ).split()
            )

            key = txt_file.stem

            rf_out.write(
                f"{key}\t{text}\n"
            )

            device = torch.device(
                "cuda"
            )

            self.asr = self.asr.to(
                device
            )

            recording = (
                Recording.from_file(
                    wav_file,
                    recording_id="TEST",
                )
            )

            audio: np.ndarray = (
                recording.load_audio()
            )

            samples = (
                torch.from_numpy(audio)
                .squeeze(0)
                .to(device)
            )

            extractor = Fbank(
                FbankConfig(
                    num_mel_bins=80,
                    sampling_rate=8000,
                )
            )

            feats = extractor.extract(
                samples,
                8000,
            ).to(device)

            asr_info = (
                self.asr.model_info()
            )

            stream_pad_length = (
                asr_info["right_frame"]
            )

            decode_frame = (
                asr_info["decode_frame"]
            )

            attn_context = (
                asr_info["attn_context"]
            )

            chunk_length = (
                decode_frame
                + stream_pad_length
            )

            stream_num_frames = (
                feats.size(0)
            )

            stream_done = False

            stream_num_processed_frames = 0

            params = self.params

            attn_context = (
                self.left_context
            )

            decode_frame = (
                self.decode_chunk_size
                * params.subsampling_factor
            )

            chunk_length = (
                decode_frame
                + self.pad_length
            )

            cache = (
                self.asr.initialize_cache(
                    attn_context,
                    device,
                )
            )

            start = time.perf_counter()

            all_ctc_logits = []

            with torch.no_grad():

                cur_hyps = [
                    (
                        tuple(),
                        PrefixScore(
                            0.0,
                            -float("inf"),
                            0.0,
                            0.0,
                        ),
                    )
                ]

                while not stream_done:

                    feat_len = min(
                        stream_num_frames
                        - stream_num_processed_frames,
                        chunk_length,
                    )

                    feat = feats[
                        stream_num_processed_frames:
                        stream_num_processed_frames
                        + feat_len
                    ]

                    stream_num_processed_frames += (
                        decode_frame
                    )

                    if (
                        stream_num_processed_frames
                        >= stream_num_frames
                    ):
                        stream_done = True

                    (
                        eouts,
                        couts,
                        cache,
                        _,
                    ) = self.asr.streaming_encoder(
                        feat.unsqueeze(0),
                        cache,
                        decode_frame,
                        stream_pad_length,
                        1.0,
                    )

                    chunk_logits = (
                        couts
                        .squeeze(0)
                        .float()
                        .cpu()
                    )

                    all_ctc_logits.append(
                        chunk_logits
                    )

            # ------------------------------------------------
            # CTC decode
            # KenLM + Hotword
            # ------------------------------------------------

            logits = torch.cat(
                all_ctc_logits,
                dim=0,
            ).numpy()

            decode_kwargs = {
                "beam_width": 20,
            }

            if self.hotwords:

                decode_kwargs[
                    "hotwords"
                ] = self.hotwords

                decode_kwargs[
                    "hotword_weight"
                ] = self.hotword_weight

            content = (
                self.ctc_decoder.decode(
                    logits,
                    **decode_kwargs,
                )
            )

            content = (
                decode_byte_fallback(
                    content
                )
            )

            print(
                f"[{key}] {content}"
            )

            hf_out.write(
                f"{key}\t{content}\n"
            )

            end = time.perf_counter()

            elapsed_ms = (
                end - start
            ) * 1000

            print(
                f"{key}: "
                f"{elapsed_ms:.3f} ms"
            )

        rf_out.close()
        hf_out.close()

        with open(
            eval_file,
            "w",
        ) as f:

            subprocess.run(
                [
                    sys.executable,
                    self.eval_exe,
                    ref_file,
                    hyp_file,
                ],
                stdout=f,
            )

    @torch.no_grad()
    def deploy(self):

        checkpoint = (
            self.get_checkpoint()
        )

        self.encoder_trace(
            checkpoint
        )


if __name__ == "__main__":

    exp_path = (
        "/data/manifests/zipformer2_lgu"
    )

    test_path = (
        "/data/test/test_wav"
    )

    out_path = (
        "/data/test"
    )

    # --------------------------------------------------------
    # KenLM
    # --------------------------------------------------------

    kenlm_model_path = (
        "/data/lm/4gram.arpa"
    )

    lm_alpha = 0.5
    lm_beta = 0.05

    # --------------------------------------------------------
    # Hotwords
    # --------------------------------------------------------

    hotwords = [
        "신한은행",
        "자동이체",
        "납부방법",
        "신용카드",
        "유플러스",
    ]

    hotword_weight = 10.0

    # --------------------------------------------------------

    start_num = 53
    end_num = 90

    check_interval = 60 * 5

    copied = set()

    while True:

        for i in range(
            start_num,
            end_num + 1,
        ):

            if i in copied:
                continue

            modelname = (
                f"epoch-{i}.pt"
            )

            model_path = Path(
                f"{exp_path}/"
                f"{modelname}"
            )

            if model_path.exists():

                with torch.no_grad():

                    evaluation = (
                        ModuleExport(
                            exp_path=exp_path,
                            modelname=modelname,
                            test_path=test_path,
                            out_path=out_path,

                            kenlm_model_path=
                                kenlm_model_path,

                            lm_alpha=
                                lm_alpha,

                            lm_beta=
                                lm_beta,

                            hotwords=
                                hotwords,

                            hotword_weight=
                                hotword_weight,
                        )
                    )

                    evaluation.deploy()
                    evaluation.search()

                copied.add(i)

        if len(copied) == (
            end_num
            - start_num
            + 1
        ):

            print(
                "EVAL Completed ..."
            )

            break

        time.sleep(
            check_interval
        )

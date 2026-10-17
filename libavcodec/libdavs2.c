/*
 * AVS2 decoding using the davs2 library
 *
 * Copyright (C) 2018 Yiqun Xu, <yiqun.xu@vipl.ict.ac.cn>
 *                    Falei Luo, <falei.luo@gmail.com>
 *                    Huiwen Ren, <hwrenx@gmail.com>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "config.h"
#include "libavutil/cpu.h"
#include "libavutil/thread.h"
#include "avcodec.h"
#include "codec_internal.h"
#include "avs2.h"
#include "decode.h"
#include "davs2.h"
#if HAVE_DAVS2_10
#include "davs2_10.h"
#endif

typedef struct DAVS2API {
    void *(*open)(davs2_param_t *param);
    void (*close)(void *decoder);
    int (*send_packet)(void *decoder, davs2_packet_t *packet);
    int (*recv_frame)(void *decoder, davs2_seq_info_t *seq, davs2_picture_t *pic);
    int (*flush)(void *decoder, davs2_seq_info_t *seq, davs2_picture_t *pic);
    void (*frame_unref)(void *decoder, davs2_picture_t *pic);
} DAVS2API;

/* Lazy opens run outside avcodec_open2's non-thread-safe init lock. davs2
 * initializes its process-wide primitive tables during its first open. */
static AVMutex davs2_open_mutex = AV_MUTEX_INITIALIZER;

static const DAVS2API davs2_api_8 = {
    davs2_decoder_open, davs2_decoder_close,
    davs2_decoder_send_packet, davs2_decoder_recv_frame,
    davs2_decoder_flush, davs2_decoder_frame_unref,
};

#if HAVE_DAVS2_10
static const DAVS2API davs2_api_10 = {
    lav_davs2_10_decoder_open, lav_davs2_10_decoder_close,
    lav_davs2_10_decoder_send_packet, lav_davs2_10_decoder_recv_frame,
    lav_davs2_10_decoder_flush, lav_davs2_10_decoder_frame_unref,
};
#endif

typedef struct DAVS2Context {
    void *decoder;
    const DAVS2API *api;
    AVPacket *pkt;
    AVPacket *submitted_props;
    int bit_depth;
    int switching;
    int draining;
    davs2_param_t    param;      // decoding parameters
    davs2_picture_t  out_frame;  // output data, frame data
    davs2_seq_info_t headerset;  // output data, sequence header
}DAVS2Context;

/* The parser supplies complete start-code units. Select by encoding precision,
 * since Main10 may also carry an internally 8-bit stream. Pixel format is not
 * available at decoder initialization and sample precision describes output. */
static int davs2_sequence_bit_depth(AVCodecContext *avctx,
                                  const uint8_t *buf, int size)
{
    int i;

    for (i = 0; i <= size - 4; i++) {
        int profile, sample_precision, encoding_precision;

        if (buf[i] || buf[i + 1] || buf[i + 2] != 1 ||
            buf[i + 3] != AVS2_SEQ_START_CODE)
            continue;
        if (size - i < 11)
            return AVERROR_INVALIDDATA;
        profile = buf[i + 4];
        if (profile != AVS2_PROFILE_MAIN && profile != AVS2_PROFILE_MAIN_PIC &&
            profile != AVS2_PROFILE_MAIN10) {
            av_log(avctx, AV_LOG_ERROR, "Unsupported AVS2 profile 0x%x.\n", profile);
            return AVERROR_PATCHWELCOME;
        }
        sample_precision = buf[i + 10] >> 5;
        encoding_precision = profile == AVS2_PROFILE_MAIN10 ?
                             (buf[i + 10] >> 2) & 7 : 1;
        if (!sample_precision || !encoding_precision)
            return AVERROR_INVALIDDATA;
        if (sample_precision > 2 || encoding_precision > 2) {
            av_log(avctx, AV_LOG_ERROR, "Unsupported AVS2 sample precision %d/%d.\n",
                   sample_precision, encoding_precision);
            return AVERROR_PATCHWELCOME;
        }
        if (sample_precision > encoding_precision) {
            av_log(avctx, AV_LOG_ERROR, "AVS2 output precision exceeds encoding precision.\n");
            return AVERROR_PATCHWELCOME;
        }
        return encoding_precision == 2 ? 10 : 8;
    }
    return 0;
}

static void davs2_close_decoder(DAVS2Context *cad)
{
    if (cad->decoder)
        cad->api->close(cad->decoder);
    cad->decoder = NULL;
    cad->api = NULL;
    cad->bit_depth = 0;
    memset(&cad->headerset, 0, sizeof(cad->headerset));
    memset(&cad->out_frame, 0, sizeof(cad->out_frame));
}

static int davs2_open_decoder(AVCodecContext *avctx, int bit_depth)
{
    DAVS2Context *cad = avctx->priv_data;

    cad->api = &davs2_api_8;
    if (bit_depth == 10) {
#if HAVE_DAVS2_10
        cad->api = &davs2_api_10;
#else
        av_log(avctx, AV_LOG_ERROR, "AVS2 10-bit decoding requires a dual-depth libdavs2 build.\n");
        return AVERROR(ENOSYS);
#endif
    }
    ff_mutex_lock(&davs2_open_mutex);
    cad->decoder = cad->api->open(&cad->param);
    ff_mutex_unlock(&davs2_open_mutex);

    if (!cad->decoder) {
        av_log(avctx, AV_LOG_ERROR, "Cannot create AVS2 %d-bit decoder.\n", bit_depth);
        return AVERROR_EXTERNAL;
    }
    cad->bit_depth = bit_depth;
    av_log(avctx, AV_LOG_VERBOSE, "Created AVS2 %d-bit decoder.\n", bit_depth);
    return 0;
}

static av_cold int davs2_init(AVCodecContext *avctx)
{
    DAVS2Context *cad = avctx->priv_data;
    int cpu_flags = av_get_cpu_flags();

    cad->param.threads = avctx->thread_count;
    cad->param.info_level = 0;
    cad->param.disable_avx = !(cpu_flags & AV_CPU_FLAG_AVX &&
                               cpu_flags & AV_CPU_FLAG_AVX2);
    cad->pkt = av_packet_alloc();
    cad->submitted_props = av_packet_alloc();
    if (!cad->pkt || !cad->submitted_props) {
        av_packet_free(&cad->pkt);
        av_packet_free(&cad->submitted_props);
        return AVERROR(ENOMEM);
    }
    return 0;
}

static int davs2_update_header(AVCodecContext *avctx)
{
    DAVS2Context *cad = avctx->priv_data;
    davs2_seq_info_t *seq = &cad->headerset;
    int ret = ff_set_dimensions(avctx, seq->width, seq->height);

    if (ret < 0)
        return ret;
    avctx->has_b_frames = FFMAX(avctx->has_b_frames, !seq->low_delay);
    if (seq->frame_rate_id < 16)
        avctx->framerate = ff_avs2_frame_rate_tab[seq->frame_rate_id];
    return 0;
}

static int davs2_dump_frame(AVCodecContext *avctx, AVFrame *frame)
{
    DAVS2Context *cad    = avctx->priv_data;
    davs2_picture_t *pic = &cad->out_frame;
    int bytes_per_sample = pic->bytes_per_sample;
    int plane, line, ret;

    if (pic->num_planes != 3 || (pic->bit_depth != 8 && pic->bit_depth != 10) ||
        bytes_per_sample != (pic->bit_depth > 8 ? 2 : 1))
        return AVERROR_INVALIDDATA;

    avctx->pix_fmt = pic->bit_depth == 10 ? AV_PIX_FMT_YUV420P10 : AV_PIX_FMT_YUV420P;
    ret = ff_get_buffer(avctx, frame, 0);
    if (ret < 0)
        return ret;
    ret = ff_decode_frame_props_from_pkt(avctx, frame, cad->submitted_props);
    if (ret < 0)
        return ret;

    switch (pic->type) {
    case DAVS2_PIC_I:
    case DAVS2_PIC_G:
        frame->pict_type = AV_PICTURE_TYPE_I;
        frame->flags |= AV_FRAME_FLAG_KEY;
        break;
    case DAVS2_PIC_P:
    case DAVS2_PIC_S:
        frame->pict_type = AV_PICTURE_TYPE_P;
        break;
    case DAVS2_PIC_B:
        frame->pict_type = AV_PICTURE_TYPE_B;
        break;
    case DAVS2_PIC_F:
        frame->pict_type = AV_PICTURE_TYPE_S;
        break;
    default:
        av_log(avctx, AV_LOG_ERROR, "Decoder error: unknown frame type\n");
        return AVERROR_EXTERNAL;
    }

    for (plane = 0; plane < 3; ++plane) {
        int width = plane ? AV_CEIL_RSHIFT(frame->width, 1) : frame->width;
        int height = plane ? AV_CEIL_RSHIFT(frame->height, 1) : frame->height;
        int size_line = width * bytes_per_sample;

        if (!pic->planes[plane] || pic->widths[plane] != width ||
            pic->lines[plane] != height || pic->strides[plane] < size_line)
            return AVERROR_INVALIDDATA;

        for (line = 0; line < pic->lines[plane]; ++line)
            memcpy(frame->data[plane] + line * frame->linesize[plane],
                   pic->planes[plane] + line * pic->strides[plane],
                   size_line);
    }

    frame->pts = pic->pts;
    frame->pkt_dts = pic->dts;
    if (pic->b_decode_error)
        frame->flags |= AV_FRAME_FLAG_CORRUPT;
    return 0;
}

static void davs2_flush(AVCodecContext *avctx)
{
    DAVS2Context *cad = avctx->priv_data;

    /* The library flush API enters EOS draining permanently. Recreate it after
     * seeking instead, and let the next sequence select the appropriate API. */
    davs2_close_decoder(cad);
    av_packet_unref(cad->pkt);
    av_packet_unref(cad->submitted_props);
    cad->switching = cad->draining = 0;
}

static av_cold int davs2_end(AVCodecContext *avctx)
{
    DAVS2Context *cad = avctx->priv_data;

    davs2_close_decoder(cad);
    av_packet_free(&cad->pkt);
    av_packet_free(&cad->submitted_props);
    return 0;
}

static int davs2_send_packet(DAVS2Context *cad, const uint8_t *data, int size,
                             int64_t pts, int64_t dts)
{
    davs2_packet_t packet = { data, size, pts, dts };

    return cad->api->send_packet(cad->decoder, &packet) == DAVS2_ERROR ?
           AVERROR_INVALIDDATA : 0;
}

static int davs2_receive_frame(AVCodecContext *avctx, AVFrame *frame)
{
    DAVS2Context *cad = avctx->priv_data;
    int ret, bit_depth, from_extradata;

    for (;;) {
        if (cad->decoder) {
            ret = (cad->switching || cad->draining ? cad->api->flush : cad->api->recv_frame)
                  (cad->decoder, &cad->headerset, &cad->out_frame);
            if (ret == DAVS2_ERROR)
                return AVERROR_EXTERNAL;
            if (ret == DAVS2_GOT_HEADER) {
                ret = davs2_update_header(avctx);
                if (ret < 0)
                    return ret;
                continue;
            }
            if (ret == DAVS2_GOT_FRAME) {
                ret = davs2_dump_frame(avctx, frame);
                cad->api->frame_unref(cad->decoder, &cad->out_frame);
                if (ret < 0)
                    av_frame_unref(frame);
                return ret;
            }
            if (cad->switching) {
                davs2_close_decoder(cad);
                cad->switching = 0;
            } else if (cad->draining) {
                return AVERROR_EOF;
            }
        } else if (cad->draining) {
            return AVERROR_EOF;
        }

        if (!cad->pkt->size) {
            ret = ff_decode_get_packet(avctx, cad->pkt);
            if (ret == AVERROR_EOF) {
                cad->draining = 1;
                continue;
            }
            if (ret < 0)
                return ret;
            if (!cad->pkt->size)
                return AVERROR(EAGAIN);
        }

        bit_depth = davs2_sequence_bit_depth(avctx, cad->pkt->data, cad->pkt->size);
        if (bit_depth < 0) {
            av_packet_unref(cad->pkt);
            return bit_depth;
        }
        if (cad->decoder && bit_depth && bit_depth != cad->bit_depth) {
            /* Retain this packet until every delayed frame from the previous
             * sequence has been returned. The legacy video decode callback
             * cannot return frames without also consuming the input packet. */
            cad->switching = 1;
            continue;
        }
        if (!cad->decoder) {
            from_extradata = !bit_depth;
            if (from_extradata)
                bit_depth = davs2_sequence_bit_depth(avctx, avctx->extradata,
                                                    avctx->extradata_size);
            if (bit_depth <= 0) {
                av_packet_unref(cad->pkt);
                if (bit_depth < 0)
                    return bit_depth;
                /* A seek may initially deliver pictures without a sequence. */
                continue;
            }
            ret = davs2_open_decoder(avctx, bit_depth);
            if (ret >= 0 && from_extradata)
                ret = davs2_send_packet(cad, avctx->extradata, avctx->extradata_size,
                                       AV_NOPTS_VALUE, AV_NOPTS_VALUE);
            if (ret < 0) {
                av_packet_unref(cad->pkt);
                davs2_close_decoder(cad);
                return ret;
            }
        }
        ret = davs2_send_packet(cad, cad->pkt->data, cad->pkt->size,
                               cad->pkt->pts, cad->pkt->dts);
        if (ret >= 0) {
            /* A pending sequence packet must not supply duration or side data
             * to delayed frames from the previous backend. Keep the existing
             * latest-submitted-packet policy within each sequence. */
            av_packet_unref(cad->submitted_props);
            ret = av_packet_copy_props(cad->submitted_props, cad->pkt);
        }
        av_packet_unref(cad->pkt);
        if (ret < 0)
            return ret;
    }
}

const FFCodec ff_libdavs2_decoder = {
    .p.name         = "libdavs2",
    CODEC_LONG_NAME("libdavs2 AVS2-P2/IEEE1857.4"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_AVS2,
    .priv_data_size = sizeof(DAVS2Context),
    .init           = davs2_init,
    .close          = davs2_end,
    FF_CODEC_RECEIVE_FRAME_CB(davs2_receive_frame),
    .flush          = davs2_flush,
    .p.capabilities = AV_CODEC_CAP_DR1 | AV_CODEC_CAP_DELAY | AV_CODEC_CAP_OTHER_THREADS,
    .caps_internal  = FF_CODEC_CAP_NOT_INIT_THREADSAFE |
                      FF_CODEC_CAP_SETS_FRAME_PROPS |
                      FF_CODEC_CAP_AUTO_THREADS,
    .p.wrapper_name = "libdavs2",
};

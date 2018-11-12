//
//  jhBarGraphLayer.swift
//  bridge8
//
//  Created by Junho HA on 2022. 2. 22.
//  Copyright © 2022년 eoflow. All rights reserved.
//

import UIKit

class jhBarGraphLayer<T> : jhCommonDataLayer<T> {
    
    override func draw(in ctx: CGContext) {
//        //        worldEllipse(context: mContext, 100, 100, 100, 100, 2, UIColor.blue.cgColor)
//        panelID = 0
//
//        var pointCloud = Array<CGPoint>()
//        var fx, fy : CGFloat
//
//        var x : Int = 0
//
//        mValuesOfDatas.removeAll()
//
//        for y in 0..<jhDataCenter.mDatas[panelID]!.d.count {
//            mValuesOfDatas.append(jhDataCenter.mDatas[panelID]!.d[y].y)
//        }
//
//        for y in mValuesOfDatas { //TODO:
//            //ref:drawLine(CGFloat(x)*axisDistance + mMargin, mMargin, CGFloat(x) * axisDistance + mMargin, 10000-mMargin)
//            x += 1
//            fx = CGFloat(x)*xDistance
//            fy = CGFloat(y)*mVerticalRatioToDraw_view + mMargin
//            drawEllipse2(ctx, fx, fy, 2, 2, thickness: 2, UIColor.blue.cgColor)
//            pointCloud.append(CGPoint.init(x: getX(fx+mMargin)!, y: getY(fy)!))
//        }
//
//        ctx.move(to: CGPoint.init(x: 0, y: 0))
//        ctx.setStrokeColorSpace(CGColorSpaceCreateDeviceRGB())
//        ctx.setStrokeColor(UIColor.blue.cgColor)
//        ctx.setLineWidth(1.0)
//        ctx.addLines(between: pointCloud)
//        ctx.strokePath()
//
//        for x in pointCloud {
//
////            ctx.move(to: CGPoint.init(x: 0, y: 0))
////
//            ctx.setFillColor(jhDraw.jhColor(r: 184, g: 70, b: 201, a: 0.5))
//            ctx.setStrokeColor(jhDraw.jhColor(r: 184, g: 70, b: 201, a: 1.0))
//            ctx.setLineWidth(1)
//
//            let rectangle = CGRect(x: x.x-5, y: x.y, width: 10, height: -(x.y-getY(mMargin)!)) //TODO: 좌표 계산 부분 한 곳으로 몰기.
////            print("current ", x)
//            ctx.addRect(rectangle)
//            ctx.drawPath(using: .fillStroke)
//        }
    }
}


//
//  jhType1graphLayer.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 25..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhType1graphLayer<T> : jhCommonDataLayer<T> {
    
    override func draw(in ctx: CGContext) {
        
        let ctime = (self.superScene as? jhSceneTimeLine)?.currentTime
        let etime = (self.superScene as? jhSceneTimeLine)?.endTime
        
        print("ctime \(ctime)")
        print(etime)
        
        panelID = 0
        
        var pointCloud = Array<CGPoint>()
        var fx, fy : CGFloat
        
        var x : Int = 0
        
        mValuesOfDatas.removeAll()
        
        for y in jhDataCenter.nonNetworkData {
            mValuesOfDatas.append(y)
        }
        
        for y in mValuesOfDatas { //TODO:
            //ref:drawLine(CGFloat(x)*axisDistance + mMargin, mMargin, CGFloat(x) * axisDistance + mMargin, 10000-mMargin)
            x += 1
            fx = CGFloat(x)*100 //TODO:
            fy = CGFloat(y)*mVerticalRatioToDraw_view + mMargin
            drawEllipse2(ctx, fx, fy, 2, 2, thickness: 2, UIColor.blue.cgColor)
            pointCloud.append(CGPoint.init(x: getX(fx+mMargin)!, y: getY(fy)!))
        }
        
        ctx.move(to: CGPoint.init(x: 0, y: 0))
        ctx.setStrokeColorSpace(CGColorSpaceCreateDeviceRGB())
        ctx.setStrokeColor(UIColor.blue.cgColor)
        ctx.setLineWidth(1.0)
        ctx.addLines(between: pointCloud)
        ctx.strokePath()
        
        for x in pointCloud {
            
            //            ctx.move(to: CGPoint.init(x: 0, y: 0))
            //
            ctx.setFillColor(jhDraw.jhColor(r: 184, g: 70, b: 201, a: 0.5))
            ctx.setStrokeColor(jhDraw.jhColor(r: 184, g: 70, b: 201, a: 1.0))
            ctx.setLineWidth(1)
            
            let rectangle = CGRect(x: x.x-5, y: x.y, width: 10, height: -(x.y-getY(mMargin)!)) //TODO: 좌표 계산 부분 한 곳으로 몰기.
            //            print("current ", x)
            ctx.addRect(rectangle)
            ctx.drawPath(using: .fillStroke)
        }
    }
}
